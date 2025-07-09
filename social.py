import json
import pyaudio
import numpy as np
import whisper
import time
from pathlib import Path
from llama_index.llms.ollama import Ollama
import warnings

# 禁用Whisper的FP16警告
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")


class SocialAgent:
    def __init__(self, profile_path, output_dir="conversation_logs"):
        self.profile = self._load_profile(profile_path)
        self.llm = Ollama(
            base_url="http://localhost:11435",
            model="gemma3:4b",
            request_timeout=180.0,
            temperature=0.8
        )
        self.output_dir = Path(output_dir)
        self._init_speech_engine()
        self._setup_files()
        self.dialogue_history = []
        self.trust_built = False

        # 优化音频参数
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.SILENCE_THRESH = 0.03
        self.MIN_SPEECH_DURATION = 1.2
        self.MAX_SPEECH_DURATION = 25
        self.CONSECUTIVE_SILENCE = 3

    def _load_profile(self, path):
        """加载个人资料文件"""
        try:
            with open(path, 'r', encoding='gbk') as f:
                return f.read().strip()
        except Exception as e:
            print(f"\033[31m[错误] 文件读取失败: {str(e)}\033[0m")
            return ""

    def _init_speech_engine(self):
        """初始化语音组件"""
        self.whisper_model = whisper.load_model("small")
        self.audio = pyaudio.PyAudio()

    def _setup_files(self):
        """初始化日志系统"""
        self.output_dir.mkdir(exist_ok=True)
        self.transcript_file = self.output_dir / "dialogue.log"
        self.audio_log_file = self.output_dir / "user_audio.wav"
        with open(self.transcript_file, 'w', encoding='utf-8') as f:
            f.write("=== 对话开始 ===\n\n")

    def _save_to_file(self, content):
        """保存文本日志"""
        with open(self.transcript_file, 'a', encoding='utf-8') as f:
            f.write(f"{content}\n\n")

    def _record_audio(self):
        """智能语音录制（带实时反馈）"""
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        frames = []
        speech_start = None
        silent_counter = 0
        recording = False

        print("\033[36m[系统] 请开始说话...\033[0m")

        try:
            while True:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Check if audio_data is valid before calculating RMS
                if audio_data.size > 0:
                    # Filter out invalid values
                    valid_audio_data = audio_data[audio_data >= 0]
                    if valid_audio_data.size > 0:
                        # Calculate mean of squared values
                        mean_squared = np.mean(np.square(valid_audio_data))
                        if mean_squared >= 0:
                            rms = np.sqrt(mean_squared)
                        else:
                            rms = 0  # Handle negative mean_squared (should not happen)
                    else:
                        rms = 0
                else:
                    rms = 0

                if rms > self.SILENCE_THRESH:
                    silent_counter = 0
                    if not recording:
                        speech_start = time.time()
                        recording = True
                        print("\033[33m[录音中]", end="", flush=True)
                    frames.append(data)
                else:
                    silent_counter += 1
                    if recording:
                        print("▋", end="", flush=True)

                if recording:
                    duration = time.time() - speech_start
                    if silent_counter >= self.CONSECUTIVE_SILENCE and duration > self.MIN_SPEECH_DURATION:
                        break
                    if duration > self.MAX_SPEECH_DURATION:
                        break

        finally:
            stream.stop_stream()
            stream.close()
            if frames and (time.time() - speech_start) >= self.MIN_SPEECH_DURATION:
                print("\033[32m [完成]\033[0m")
                return b''.join(frames)
            return None

    def _transcribe_audio(self, audio_bytes):
        """语音转文字（优化提示词）"""
        if not audio_bytes:
            return ""

        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # 优化后的参数配置
            result = self.whisper_model.transcribe(
                audio_np,
                language='zh',
                temperature=0.0,  # 降低随机性提高一致性
                best_of=1,  # 减少候选数量加快速度
                beam_size=3,  # 适中的beam size平衡速度和质量
                patience=1.0,  # 减少patience加快解码
                initial_prompt="这是一段普通话对话，请准确转写，不要添加未说出的内容",
                no_speech_threshold=0.6,  # 提高无语音检测阈值
                logprob_threshold=-0.5,  # 提高log概率阈值
                compression_ratio_threshold=2.0  # 控制压缩率避免无意义输出
            )

            # 后处理优化
            text = result["text"].strip()
            # 移除常见的识别错误模式
            for pattern in ["谢谢", "感谢", "呃", "啊"]:
                text = text.replace(pattern, "")
            return text if text else "[未识别到有效语音]"

        except Exception as e:
            print(f"\033[31m[错误] 转写失败: {str(e)}\033[0m")
            return "[语音转写失败]"

    def _generate_opener(self):
        """生成自然流畅的开场问题"""
        prompt = f"""根据用户信息生成一个自然流畅的中文开场白，不超过30字，直接以问题或话题开始：
        用户档案：{self.profile}
        要求：
        1. 包含学校、专业或奖项等具体信息
        2. 语气亲切，像是朋友聊天
        3. 引发对方兴趣并便于后续对话
        示例：'听说你在武汉读人工智能硕士，还拿过不少英语竞赛奖，平时怎么平衡学习和比赛的？'
        """
        response = self._get_llm_response(prompt)
        # 清理可能的多余前缀（如“问题：”）
        return response.split("：")[-1].strip() if "：" in response else response

    def _generate_followup(self):
        """生成上下文相关的追问"""
        context = "\n".join(self.dialogue_history[-3:])
        prompt = f"""根据最近对话生成简短的自然追问：
        当前对话：{context}
        要求：
        1. 延续当前话题
        2. 包含用户提到的具体细节
        3. 直接提问不要前缀"""
        return self._get_llm_response(prompt)

    def _generate_trust(self):
        """生成信任建立内容"""
        context = "\n".join(self.dialogue_history)
        prompt = f"""基于对话历史生成深入建立信任：
        完整对话：{context}
        要求：
        1. 结合用户提到的专业技能
        2. 提出具体合作场景
        3. 直接说明不要前缀"""
        return self._get_llm_response(prompt)

    def _get_llm_response(self, prompt):
        """优化响应处理"""
        try:
            response = self.llm.complete(prompt).text.strip()
            # 清理响应格式
            response = response.split("\n")[0].replace("**", "").replace("问题：", "").strip()
            if len(response) < 3:  # 处理空响应
                return "能多聊聊这个话题吗？"
            return response
        except Exception as e:
            print(f"\033[31m[错误] 生成失败: {str(e)}\033[0m")
            return "能详细说说吗？"

    def _process_interaction(self, question):
        """处理单次交互（优化显示格式）"""
        print(f"\033[34m[Agent] {question}\033[0m")
        self._save_to_file(f"Agent: {question}")
        self.dialogue_history.append(f"Agent: {question}")

        # 获取用户回复
        audio_data = self._record_audio()
        if not audio_data:
            return False

        user_text = self._transcribe_audio(audio_data)
        if any(w in user_text for w in ["结束", "停止", "退出"]):
            return False

        if user_text:
            print(f"\033[32m[User] {user_text}\033[0m")
            self._save_to_file(f"User: {user_text}")
            self.dialogue_history.append(f"User: {user_text}")
            return True
        return False

    def run(self):
        """优化后的对话流程"""
        print("\n\033[36m=== 对话启动 ===\033[0m")

        # 生成首个问题
        opener = self._generate_opener()
        if not self._process_interaction(opener):
            return

        # 动态对话循环
        while len(self.dialogue_history) < 8:  # 控制对话轮次
            # 每2轮普通对话后插入信任建立
            if len(self.dialogue_history) % 2 == 0 and not self.trust_built:
                trust_question = self._generate_trust()
                self.trust_built = True
                if not self._process_interaction(trust_question):
                    break
            else:
                followup = self._generate_followup()
                if not self._process_interaction(followup):
                    break

        print("\033[33m[系统] 对话日志已保存至:", self.transcript_file, "\033[0m")


if __name__ == "__main__":
    file_number = input("请输入用户档案编号（1-9）：").strip()
    txt_path = f"E:\\data\\{file_number}.txt"

    try:
        agent = SocialAgent(txt_path)
        agent.run()
    except Exception as e:
        print(f"\033[31m[错误] 系统异常: {str(e)}\033[0m")