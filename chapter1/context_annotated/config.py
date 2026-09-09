# -*- coding: utf-8 -*-
# ↑ 编码声明。Python 3 默认就是 UTF-8，写不写都行；
#   写上是为了告诉编辑器和老工具「这个文件里有中文，按 UTF-8 读」。

"""
Configuration module for Context-Aware Agent
↑ 这是「模块文档字符串（module docstring）」，必须放在文件最顶部（import 之前）。
  它不是注释，而是一个真正的字符串对象，Python 会把它存到模块的 __doc__ 属性里。
  作用：用 help(模块名) 或 IDE 悬浮提示时能看到这段说明。
  和注释的区别：# 开头的注释会被解释器直接忽略，docstring 会被保留下来。
"""

# ─────────────────────────────────────────────────────────────
# 第一部分：导入（import）
# ─────────────────────────────────────────────────────────────

import os  # 标准库：和操作系统打交道。这里主要用 os.getenv() 读环境变量、os.makedirs() 建目录

from typing import Optional  # typing 模块提供「类型提示」用的类型
# Optional[str] 等价于 str | None，意思是「这个值要么是字符串，要么是 None」
# 写成 Optional[str] 是为了兼容 Python 3.9 及以下（3.10+ 才能用 | 语法）

from dotenv import load_dotenv  # 第三方库 python-dotenv：把 .env 文件里的键值对读进环境变量
# 为什么需要？因为 API Key 不能写死在代码里（会泄露），
# 通常放在项目根目录的 .env 文件中，并且要把 .env 加进 .gitignore

# Load environment variables
load_dotenv()  # ↑ 执行这一句，.env 文件里的配置就"注入"到 os.environ 了
# 之后 os.getenv("XXX_API_KEY") 才能读到 .env 里写的值
# 注意：load_dotenv() 是「导入时执行」的副作用——任何人 import config 都会自动加载一次


# ─────────────────────────────────────────────────────────────
# 第二部分：模块级辅助函数
# ─────────────────────────────────────────────────────────────

def _reasoning_safe_temperature(model, requested=1.0):
    """Reasoning models (Kimi K3, GPT-5, ...) only accept temperature=1.
    Return 1 for those; otherwise the requested value so non-reasoning
    providers (Doubao, DeepSeek, older Moonshot) are unchanged.
    ↑ 这是「函数文档字符串（function docstring）」，说明这个函数干什么。
      用三引号包裹，可以跨多行。
    """
    # ↑ 函数名以单下划线 _ 开头：这是 Python 的命名约定，表示「内部私有函数」
    #   不是语法强制，只是告诉别人「这是模块内部用的，外部别直接调用」

    # ↓ 这一行做了三件事（从右往左看）：
    #   1. str(model or "")  —— model 为空/None 时变成空字符串，避免报错（短路求值）
    #   2. .lower()          —— 统一转小写，方便后面做包含判断
    #   3. .replace("/", "-") —— 把斜杠换成横杠，因为模型 id 可能写成 "moonshot/kimi-k3"
    m = str(model or "").lower().replace("/", "-")

    # ↓ 三元表达式：条件成立返回前面的值，否则返回后面的
    #   如果模型名里含 "kimi-k3" 或 "gpt-5"（推理型模型），强制温度=1
    #   否则原样返回调用者要求的值（默认 1.0）
    # 背景知识：temperature 控制生成随机性，0=确定性输出，越高越发散。
    #           推理型模型（thinking model）内部已有固定解码策略，API 只接受 temperature=1。
    return 1 if ("kimi-k3" in m or "gpt-5" in m) else requested


# ─────────────────────────────────────────────────────────────
# 第三部分：导入共享的「供应商注册表」
# ─────────────────────────────────────────────────────────────

# Provider resolution lives in the shared agentbook package so every chapter
# stays consistent; see agentbook/providers.py. The fallback keeps this
# experiment runnable from a checkout where agentbook is not installed.
# ↑ 上面几行英文注释解释「为什么」：供应商解析逻辑放在共享包 agentbook 里，
#   这样每一章的代码都保持一致；下面的 except 分支是兜底，
#   万一 agentbook 没装（比如直接拷贝了这一章的代码），也能跑起来。

try:
    # ↓ 尝试正常导入。from xxx import a, b, c 可以一次导入多个名字
    from agentbook.providers import (
        PROVIDERS,  # dict[str, Provider]：供应商名 → 供应商配置对象 的注册表
        SUPPORTED_PROVIDERS,  # tuple[str, ...]：所有支持的供应商名字（含别名），用于命令行 --help 展示
        canonical_provider,  # 函数：把别名（如 "qwen"、"bailian"）归一成标准名（如 "dashscope"）
        canonical_provider as _canonical_provider,  # ↑ as 关键字：起个别名
        # 同一个函数起两个名字，是为了表达「意图」：
        #   canonical_provider  → 对外公开使用
        #   _canonical_provider → 本模块内部使用（下划线前缀 = 内部约定）
        map_model_to_openrouter,  # 函数：把裸模型 id（如 "gpt-5"）映射成 OpenRouter 格式（"openai/gpt-5"）
        resolve_backend,  # 函数：综合解析出「key + base_url + model」的最终调用配置
        resolve_llm_backend,  # 函数：旧的兼容入口，返回四元组（历史遗留）
    )

except ImportError:  # ↓ 如果上面的 import 抛 ImportError（模块找不到），走这个分支
    # pragma: no cover 是给 pytest-cov 看的标记，意思是「这行不算测试覆盖率」，
    # 因为这段只在没装 agentbook 时才执行，正常测试跑不到

    import sys as _sys  # 导入 sys 模块（Python 解释器相关的信息），别名 _sys

    # ↓ 把仓库根目录塞进模块搜索路径 sys.path
    #   __import__("pathlib") 是「动态导入」，等价于 import pathlib，但不用写在文件顶部
    #   Path(__file__)            → 当前文件 config.py 的路径
    #     .resolve()              → 转成绝对路径（把符号链接也解析掉）
    #     .parents[0]             → chapter1/context/
    #     .parents[1]             → chapter1/
    #     .parents[2]             → 仓库根目录 ai-agent-book/
    #   之所以绕这么一圈，是为了让解释器能找到根目录下的 agentbook 包
    _sys.path.insert(
        0, str(__import__("pathlib").Path(__file__).resolve().parents[2])
    )
    # insert(0, ...) 表示插到搜索路径最前面，优先级最高

    # ↓ 路径加好了，再导入一次。这次应该能成功
    from agentbook.providers import (
        PROVIDERS,
        SUPPORTED_PROVIDERS,
        canonical_provider,
        canonical_provider as _canonical_provider,
        map_model_to_openrouter,
        resolve_backend,
        resolve_llm_backend,
    )
# 【知识点】try/except 的价值：让代码在「依赖缺失」时有降级方案，而不是直接崩溃。
#   这叫「优雅降级（graceful degradation）」。


# ─────────────────────────────────────────────────────────────
# 第四部分：Config 类 —— 整个项目的「配置中心」
# ─────────────────────────────────────────────────────────────

class Config:
    """Configuration settings for the agent
    ↑ 类的 docstring。
      这个类不用来「创建对象」，而是当成一个「命名空间」用：
      所有配置项都写成「类变量」，直接通过 Config.XXX 访问，无需实例化。
      这是 Python 里很常见的配置写法（比全局变量更整洁、更好查找）。
    """

    # ── Provider Configuration（供应商配置）──────────────────
    # ↓ os.getenv(key, default)：读环境变量 key；没设置就用 default
    #   读出来后 .lower() 统一小写，避免用户在 .env 里写成 "Doubao" 导致匹配失败
    #   类型注解 ": str" 表示这个变量应该是字符串（只是提示，不强制检查）
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "doubao").lower()
    #                                              ↑ 默认用豆包（字节跳动）

    # ── API Configuration（各家 API 的密钥和地址）────────────
    # 模式完全一样：密钥从环境变量读（默认空字符串），地址写死常量

    # 阿里云百炼 / DashScope（通义千问 Qwen）
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL: str = os.getenv(
        "DASHSCOPE_BASE_URL",  # 允许用环境变量覆盖
        "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 默认值：国内站
    )
    # 为什么要允许覆盖？因为国际站的 key 必须配国际站的域名：
    #   DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    # 【知识点】这叫「配置外部化」——把可能变化的值从代码里抽出去，改配置不改代码。

    # SiliconFlow（硅基流动）
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"  # 不可覆盖，写死

    # 火山引擎 Ark（豆包 Doubao）
    ARK_API_KEY: str = os.getenv("ARK_API_KEY", "")
    ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"

    # Moonshot（月之暗面 Kimi）
    MOONSHOT_API_KEY: str = os.getenv("MOONSHOT_API_KEY", "")
    MOONSHOT_BASE_URL: str = "https://api.moonshot.cn/v1"

    # DeepSeek（深度求索）
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"  # 同样可覆盖（私有化部署时用）
    )

    # 智谱 GLM
    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")
    ZHIPU_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4"

    # ── Model Configuration（模型参数）───────────────────────
    MODEL_NAME: str = os.getenv("MODEL_NAME", "")  # 留空 = 用供应商的默认模型（见 get_default_model）
    # ↓ 注意这里的类型转换：os.getenv 永远返回字符串，得手动转成想要的类型
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))  # 转 float
    MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "1000"))  # 转 int
    # 如果 .env 里写的是乱码（比如 MODEL_TEMPERATURE=abc），这里会抛 ValueError 直接崩掉。
    # 生产代码会用 try/except 包一层给出友好提示，教学项目就省略了。

    # ── Agent Configuration（Agent 行为参数）─────────────────
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    # ↑ Agent 最多循环多少轮（每轮 = 调一次大模型 + 执行工具）。
    #   这是「防死循环」的保险丝：模型可能一直调工具停不下来，烧钱又卡死。

    # ↓ 这一行是个经典写法，拆开看：
    #   os.getenv("ENABLE_REASONING", "true")  → 读变量，默认字符串 "true"
    #   .lower()                               → 统一小写
    #   == "true"                              → 比较，得到布尔值 True/False
    # 好处：用户在 .env 里写 true / True / TRUE / yes 都能正确识别
    ENABLE_REASONING: bool = os.getenv("ENABLE_REASONING", "true").lower() == "true"

    # ── Test Configuration（测试用配置）──────────────────────
    TEST_PDF_URL: str = os.getenv(
        "TEST_PDF_URL",
        "https://www.berkshirehathaway.com/qtrly/1stqtr23.pdf"  # 默认：伯克希尔季度报
    )

    # ── Currency Configuration（汇率表）──────────────────────
    # 注意：这里没有类型注解，Python 会自动推断为 dict[str, float]
    # 这是「硬编码的示例汇率」。真实项目要调实时汇率 API（注释里也承认了这点）。
    EXCHANGE_RATES = {
        "USD": 1.0,  # 美元作为基准货币，所以是 1.0
        "EUR": 0.92,  # 1 美元 = 0.92 欧元
        "GBP": 0.79,
        "JPY": 149.50,
        "CNY": 7.24,
        "CAD": 1.36,
        "AUD": 1.53,
        "CHF": 0.88,
        "INR": 83.12,
        "SGD": 1.34,
    }
    # 【知识点】为什么用字典（dict）？因为它是「键 → 值」映射，
    #   查汇率时 EXCHANGE_RATES["CNY"] 是 O(1) 常数时间，比遍历列表快得多。

    # ── Logging Configuration（日志配置）─────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # ↑ 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
    #   设为 INFO 就只显示 INFO 及以上，DEBUG 的调试信息被过滤掉

    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    # ↑ 注意：这里没给 default，所以没设置时 os.getenv 返回 None
    #   类型注解写成 Optional[str] 就是在说「这个值可能是 None」——对上了！

    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    # ↑ logging 模块的格式化字符串，四个占位符分别是：
    #   %(asctime)s    → 时间
    #   %(levelname)s  → 级别（INFO/ERROR...）
    #   %(name)s       → logger 名字（一般是模块名）
    #   %(message)s    → 真正的日志内容

    # ── File paths（路径配置）────────────────────────────────
    RESULTS_DIR: str = "results"  # 实验结果输出目录
    TEST_PDFS_DIR: str = "fixtures/pdfs"  # 测试用 PDF 存放目录（相对路径）
    # 注意用正斜杠 / 而不是反斜杠 \：
    #   正斜杠在 Windows / Linux / macOS 上都能正常工作；
    #   写反斜杠在字符串里还得转义成 "\\"，既难看又容易出错。

    # ═══════════════════════════════════════════════════════
    # 下面是「类方法」。先搞懂三种方法的区别：
    #
    #   1. 实例方法  def f(self)        → 通过 对象.f() 调用，self 是对象本身
    #   2. 类方法    @classmethod
    #                def f(cls)         → 通过 类名.f() 调用，cls 是类本身
    #   3. 静态方法  @staticmethod
    #                def f()            → 通过 类名.f() 调用，不接收 self 也不接收 cls
    #
    # 这个类全程不实例化，所以所有方法都是 @classmethod，
    # 用 cls.XXX 就能读到上面定义的类变量。
    # ═══════════════════════════════════════════════════════

    @classmethod  # 装饰器：给下面的函数「套一层壳」，把它变成类方法
    def get_api_key(cls, provider: str = None) -> str:
        """
        Get API key for the specified provider

        Args:
            provider: Provider name (defaults to LLM_PROVIDER)

        Returns:
            API key for the provider
        ↑ Google 风格的 docstring：分 Args / Returns 段落。
          Args 写参数说明，Returns 写返回值说明。IDE 能识别这种格式做智能提示。
        """
        provider = provider or cls.LLM_PROVIDER
        # ↑ 惯用写法：「调用者没传就用默认值」
        #   原理是 or 的短路特性：provider 是 None/空字符串等「假值」时，取右边
        #   等价写法（更严谨，但啰嗦）：
        #     if provider is None:
        #         provider = cls.LLM_PROVIDER

        # The shared registry knows every provider's key variables, so this
        # stays correct as providers are added there.
        # ↑ 注释说明设计意图：不在本文件里写一堆 if/elif 判断供应商，
        #   而是去查共享的 PROVIDERS 注册表——以后新增供应商，这里不用改代码。
        #   这叫「把变化关进一个地方」（单一职责 / 开闭原则）。

        try:
            # ↓ 一行里做了三件事（从里往外读）：
            #   1. _canonical_provider(provider)  → 别名转标准名，"qwen" → "dashscope"
            #   2. PROVIDERS[...]                 → 从注册表取出 Provider 对象
            #   3. .api_key()                     → 调用它的方法，去环境变量里读密钥
            return PROVIDERS[_canonical_provider(provider)].api_key()

        except KeyError:
            # ↓ KeyError：字典里没有这个键（即「不认识的供应商」）
            #   处理策略：不抛异常，返回空字符串，让上层去判断「没 key」
            #   这叫「容错优于崩溃」
            return ""

    @classmethod
    def get_default_model(cls, provider: str = None) -> str:
        """
        Get default model for the specified provider
        """
        provider = provider or cls.LLM_PROVIDER  # 同上，没传就用全局配置
        provider = provider.lower()  # 统一小写，保证后面查表能命中

        if cls.MODEL_NAME:
            # ↑ 利用 Python 的「真值测试（truthy）」：非空字符串 = True，空字符串 = False
            #   用户在 .env 里显式指定了 MODEL_NAME，就优先用用户的，不再查表
            return cls.MODEL_NAME
        # 这个 if 没有 else，因为 if 里面直接 return 了——
        # 这叫「提前返回 / 卫语句（guard clause）」，能减少一层缩进，代码更清爽

        try:
            return PROVIDERS[_canonical_provider(provider)].default_model
            # ↑ 查注册表拿默认模型，例如 dashscope → "qwen3.7-plus"
        except KeyError:
            return ""  # 未知供应商，返回空字符串

    @classmethod
    def validate(cls, provider: str = None) -> bool:
        """
        Validate required configuration

        Returns:
            True if configuration is valid
        """
        provider = provider or cls.LLM_PROVIDER

        # resolve_backend already accounts for providers that need no key
        # (ollama) and for the OpenRouter fallback, and its error names the
        # exact variables to set -- so a missing key is not the only signal.
        # ↑ 注释在解释「为什么不自己检查 key 是否为空」：
        #   因为情况比想象复杂——
        #     · ollama 这种本地模型不需要 key
        #     · 没有 key 时可能自动走 OpenRouter 兜底
        #   而 resolve_backend() 抛出的错误信息里会直接告诉你该设置哪个环境变量，
        #   比自己 print 一句「key 缺失」有用得多。

        try:
            resolve_backend(provider)  # 尝试解析后端；配置不全时会抛 ValueError
        except ValueError as exc:  # as exc：把异常对象绑定到变量 exc，后面能读它的内容
            print(f"ERROR: {exc}")  # f-string：在字符串里用 {} 插入变量值
            print("Please set it in .env file or as environment variable")
            return False  # 校验失败

        return True  # 没抛异常 = 校验通过
        # 【知识点】try/except 的另一种用法：不用 if 判断，而是「让它做，出错再处理」
        #   这叫 EAFP 风格（Easier to Ask Forgiveness than Permission，请求宽恕比请求许可容易）
        #   相对的是 LBYL（Look Before You Leap，三思而后行）——先 if 检查再做。
        #   Python 社区普遍偏好 EAFP。

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        # ↓ os.makedirs() 创建目录（支持一次创建多级）
        #   exist_ok=True 是关键：目录已存在时**不报错**
        #   不加这个参数，第二次运行就会抛 FileExistsError
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.TEST_PDFS_DIR, exist_ok=True)

    @classmethod
    def get_model_config(cls) -> dict:
        """
        Get model configuration as dictionary
        """
        # ↓ 返回一个字典。这是「打包参数」的常见手法：
        #   调用方拿到这个 dict 后，可以用 **解包 直接传给 API：client.chat(**config)
        return {
            "model": cls.MODEL_NAME,
            # ↓ 这里调用了文件开头定义的那个内部函数：
            #   推理型模型强制 temperature=1，其他模型用配置值
            "temperature": _reasoning_safe_temperature(cls.MODEL_NAME, cls.MODEL_TEMPERATURE),
            "max_tokens": cls.MODEL_MAX_TOKENS,
            # ↑ 注意最后这个逗号（尾随逗号）。加不加都合法，
            #   加了的好处：以后在下面新增一行时，diff 只显示新增那一行，很干净。
        }
        # 返回类型注解 -> dict 表示返回字典。
        # 更精确的写法是 -> Dict[str, Any]，需要从 typing 导入 Dict、Any。

    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        provider = canonical_provider(cls.LLM_PROVIDER)  # 归一化供应商名
        api_key = cls.get_api_key(provider)  # 取密钥（但下面不会打印它！）

        print("\n" + "=" * 50)
        # ↑ "=" * 50 是字符串乘法：把 "=" 重复 50 次，得到一条分隔线
        #   "\n" 是换行符，先空一行再画分隔线，输出更好看
        #   对比：其他语言要写 for 循环或调用 repeat()，Python 一个运算符搞定

        print("CONFIGURATION")
        print("=" * 50)

        print(f"Provider: {provider}")  # f-string，Python 3.6+ 的现代写法
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Temperature: {cls.MODEL_TEMPERATURE}")
        print(f"Max Tokens: {cls.MODEL_MAX_TOKENS}")  # 注意是 MODEL_MAX_TOKENS，别漏了前缀
        print(f"Max Iterations: {cls.MAX_ITERATIONS}")

        # ↓ 这里又是一个「三元表达式」
        #   只显示「是否设置」，绝不打印密钥本身——这是安全红线
        print(f"API Key Set: {'Yes' if api_key else 'No'}")

        print(f"Log Level: {cls.LOG_LEVEL}")
        print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────────
# 【自测小实验】把本文件当脚本直接运行时会执行这里
# ─────────────────────────────────────────────────────────────
# __name__ 是 Python 自动给每个模块设的变量：
#   · 直接运行这个文件        → __name__ == "__main__"
#   · 被别人 import config   → __name__ == "config"
# 所以这个 if 里的代码「只在直接运行时执行」，import 时不会打扰别人。
# 这是给模块写「自测代码」的标准姿势。
if __name__ == "__main__":
    print("=== 自测：Config 类的用法 ===")

    print("\n[1] 直接访问类变量（不需要创建对象）")
    print(f"  供应商    : {Config.LLM_PROVIDER}")
    print(f"  最大轮数  : {Config.MAX_ITERATIONS}")
    print(f"  美元→人民币: {Config.EXCHANGE_RATES['CNY']}")

    print("\n[2] 调用类方法（用类名.方法名，不用 new）")
    Config.print_config()

    print("\n[3] 温度修正函数：推理模型 vs 普通模型")
    print(f"  kimi-k3 + 0.3  → {_reasoning_safe_temperature('kimi-k3', 0.3)}")     # 期望 1
    print(f"  doubao  + 0.3  → {_reasoning_safe_temperature('doubao-seed', 0.3)}")  # 期望 0.3

    print("\n[4] 打包模型参数")
    print(f"  {Config.get_model_config()}")
