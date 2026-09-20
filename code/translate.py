# translate.py
import os
import sys
import re
from config import API_SECRET_KEY, BASE_URL, LOCAL_MODEL_PATH, DEVICE, OLLAMA_MODEL
from utils import get_gpt_pipeline, extract_content
import time
import concurrent.futures
from functools import partial
from typing import List, Optional
from openai import OpenAI

# Initialize the OpenAI client
openai_client = OpenAI(
    api_key=API_SECRET_KEY,
    base_url=BASE_URL
)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def preprocess_and_segment_text(classical_text: str, max_segment_length: int = 2000) -> List[str]:

    cleaned_text = re.sub(r"[◎■※【】]", "", classical_text)
    segments = []

    while len(cleaned_text) > max_segment_length:
        split_pos = -1

        for pos in range(max_segment_length, max_segment_length - 100, -1):
            if pos < len(cleaned_text) and cleaned_text[pos] in ["。", "！", "？", "」", "》", "；", "，"]:
                split_pos = pos + 1
                break

        if split_pos == -1:
            split_pos = max_segment_length

        segments.append(cleaned_text[:split_pos])
        cleaned_text = cleaned_text[split_pos:]

    segments.append(cleaned_text)
    return segments


def extract_english_result(translation_response: str) -> Optional[str]:

    if not translation_response:
        return None

    if "#english:" in translation_response:
        return translation_response.split("#english:")[1].strip()
    else:
        return None


def extract_mandarin_result(translation_response: str) -> Optional[str]:
    """Parse the #mandarin: tag produced by Stage 1 of the Shanghainese pipeline."""
    if not translation_response:
        return None

    if "#mandarin:" in translation_response:
        return translation_response.split("#mandarin:")[1].strip()
    return None


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------

# Phrases that indicate the LLM is uncertain or hedging about a translation.
# Checked against the final English output so dialect-stage failures surface
# in a single pass regardless of pipeline depth.
_HEDGE_PATTERNS = [
    r"\broughly\b",
    r"\bapproximately\b",
    r"\bunclear\b",
    r"\bnot clear\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\buncertain\b",
    r"\bI['']m not sure\b",
    r"\bmay mean\b",
    r"\bmight mean\b",
    r"\bhard to translate\b",
    r"\bdifficult to translate\b",
    r"\bno direct (English )?equivalent\b",
    r"\bno English equivalent\b",
    r"\bcannot (be )?translated\b",
    r"\[unclear\b",
    r"\[uncertain\b",
    r"\(unclear\)",
    r"\(uncertain\)",
    r"translation uncertain",
    r"\bunknown (dialect )?term\b",
]

_HEDGE_RE = [re.compile(p, re.IGNORECASE) for p in _HEDGE_PATTERNS]


def quality_check(translated_text: str) -> dict:
    """
    Scan a translated English string for LLM hedging phrases that indicate
    low translation confidence.

    Returns a dict:
        {
            "flagged":    bool,
            "reasons":    list[str],   # matched phrases, deduped
            "confidence": str          # "high" | "low"
        }

    Intended use: call on every translated segment before scoring; log
    flagged results separately so dialect generation quality can be reviewed
    without polluting the main evaluation metrics.
    """
    if not translated_text:
        return {"flagged": True, "reasons": ["empty translation"], "confidence": "low"}

    matched = []
    for pattern in _HEDGE_RE:
        hit = pattern.search(translated_text)
        if hit:
            matched.append(hit.group(0))

    matched = list(dict.fromkeys(matched))  # preserve order, dedupe
    flagged = len(matched) > 0
    return {
        "flagged": flagged,
        "reasons": matched,
        "confidence": "low" if flagged else "high",
    }


# ---------------------------------------------------------------------------
# Classical Chinese pipeline  (original — unchanged)
# ---------------------------------------------------------------------------

def create_translation_prompt(segment_text: str) -> str:

    return f"""
    你是一位精通中国古典文学和英文的翻译专家。请执行以下翻译任务：

    1. 首先，将以下文言文准确翻译成现代汉语（大白话），要求**意译为主，兼顾语义与语气**, 然后，将上述现代汉语翻译成**流畅自然、符合英语表达习惯的英文句子**。
    「{segment_text}」

    2. 在翻译过程中，请注意以下事项：
    - **意译优先**：以传达原文含义为核心，避免机械直译；
    - 保留原文的所有含义、语气、细节与情感色彩；
    - 如有**典故或历史背景**，应根据其内涵进行意译，并在括号中提供简短解释；
    - 对于文化特定的概念，在括号中提供简短解释；
    - 不要添加额外的评论、分析或解释，只给出针对原文翻译好的内容；
    - 确保翻译结果符合学术标准，语言准确且地道；
    - 避免在英文翻译中夹杂中文词汇。

    【Few-Shot 示例】：

    示例一（哲理类 / 论语风格）：
    文言文: 子曰："温故而知新，可以为师矣。"
    #english: The Master said, "Those who reflect on the past and gain new understanding are fit to teach others."

    示例二（骈文体 / 辞章优美）：
    文言文: 山光悦鸟，水色空蒙。游目骋怀，足以极视听之娱。
    #english: The mountain light delights the birds; the waters shimmer with a soft haze. Wandering with open eyes and an unburdened heart, one may fully enjoy the pleasures of sight and sound.

    示例三（史传体 / 史记风格）：
    文言文: 项羽力能扛鼎，才气过人，然终败于垓下。
    #english: Xiang Yu possessed the strength to lift a cauldron and the talent to outshine all others, yet he was ultimately defeated at Gaixia.

    示例四（寓言体 / 先秦诸子风格）：
    文言文e: 守株待兔，冀复得兔，兔不可复得，而身为宋国笑。
    #english: He waited by the tree stump, hoping another rabbit would come running — but none ever did, and he became the laughingstock of the State of Song.

    示例五（用典 / 借古喻今）：
    文言文: 愿效老生之献策，如姜尚之垂纶。
    #english: I wish to offer my counsel like an old scholar, just as Jiang Shang (a legendary statesman who gained recognition only in old age) cast his line in still waters, waiting for fate to call him into service.

    示例六（抒情议论结合 / 唐宋散文风格）：
    文言文: 不以物喜，不以己悲。居庙堂之高则忧其民，处江湖之远则忧其君。
    #english: He does not rejoice over external things, nor grieve over personal misfortunes. When in high office, he worries for the people; when far from court, he worries for his ruler.

    请严格按照以下格式输出你的翻译的英文结果：

    #english: [你的英文翻译结果]
    """


def translate_single_segment(
    segment_text: str,
    model_type: str = "api",
    model_name: str = "deepseek-chat",
    max_tokens: int = 2000
) -> Optional[str]:

    prompt = create_translation_prompt(segment_text)

    try:
        if model_type == "api":
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional translator"},
                    {"role": "user", "content": prompt}
                ],
            )
            result = response.choices[0].message.content

        else:
            from utils import get_gpt_pipeline
            result = get_gpt_pipeline(
                text=prompt,
                model_id=model_name,
                max_tokens=max_tokens,
                model_type=model_type
            )

        return extract_english_result(result)

    except Exception as e:
        print(f"Errors in classical Chinese translation: {e}")
        return None


def serial_translate_segments(
    segments: List[str],
    model_type: str = "api",
    model_name: str = "deepseek-chat",
    max_tokens: int = 2000
) -> str:

    results = []

    for i, segment in enumerate(segments):
        print(f"Translating segment {i+1}/{len(segments)} ...")
        try:
            result = translate_single_segment(
                segment,
                model_type,
                model_name,
                max_tokens
            )

            if result is None:
                print(f"Warning: Translation of paragraph {i+1} failed, using placeholder")
                result = f"[Translation of paragraph {i+1} failed]"

            results.append(result)
        except Exception as e:
            print(f"Error: Exception occurred while translating segment {i+1}: {str(e)}")
            results.append(f"[Translation of paragraph {i+1} failed]")

    return " ".join(results)


def classical_chinese_to_english(
    classical_text: str,
    model_type: str = "api",
    model_name: str = "deepseek-chat",
    max_tokens: int = 2000
) -> Optional[str]:

    segments = preprocess_and_segment_text(classical_text)

    return serial_translate_segments(
        segments,
        model_type,
        model_name,
        max_tokens
    )


# ---------------------------------------------------------------------------
# Shanghainese pipeline  (new — two-stage: Shanghainese → Mandarin → English)
# ---------------------------------------------------------------------------

def create_shanghainese_to_mandarin_prompt(segment_text: str) -> str:
    """
    Stage 1: Shanghainese (Wu dialect) → Standard Mandarin.

    Priority order:
      1. Convey the speaker's intended meaning faithfully (meaning > literal).
      2. Flag Wu-dialect-specific terms and idioms with a parenthetical gloss
         so the meaning is not lost when Stage 2 translates to English.
      3. Preserve register (conversational stays conversational, narrative
         stays narrative, etc.).

    Output tag: #mandarin:
    """
    return f"""
    你是一位精通吴语（沪语）和普通话的翻译专家。请将以下上海话（吴语）内容翻译成标准普通话。

    翻译要求：
    - **意译优先**：以传达说话人的真实意图为核心，避免逐字对译；
    - 对于**吴语特有词汇、语气词或文化概念**，在括号内附上简短的解释性注释，例如：弄堂（上海石库门里弄小巷）、钱庄（旧上海私营传统银行）；
    - 保持原文的语气、语态和情感色彩（口语体保持口语，叙事体保持叙事）；
    - 不要添加评论或分析，只输出翻译结果；
    - 输出语言为流畅自然的标准普通话。

    【Few-Shot 示例】：

    示例一（口语对话体）：
    上海话: 阿拉今朝碰到隔壁张老伯了，伊讲弄堂里头那家钱庄要关门了，侬晓得伐？
    #mandarin: 我今天碰到隔壁的张老伯了，他说弄堂（上海石库门里弄小巷）里那家钱庄（旧上海私营传统银行）要关门了，你知道吗？

    示例二（叙事体）：
    上海话: 话说老早辰光，十六铺码头边上有个跑单帮的汉子，专门替洋行里向传递要紧文件，脚头邪气快，从来勿出差错。
    #mandarin: 话说很早以前，十六铺码头（旧上海最重要的客货两用码头）旁边有个跑单帮（独自从事小额贸易或走私的商人）的男子，专门替洋行（外国商行）里面传递重要文件，行动非常迅速，从来不出差错。

    示例三（说明指令体）：
    上海话: 要做好这桩生意，其一要先摸清行市，其二要找对路子，其三要把账算得邪气清爽，勿然容易吃亏上当。
    #mandarin: 要做好这笔生意，第一要先摸清市场行情（行市：市场价格与供需状况），第二要找对渠道和关系网络，第三要把账目算得非常清楚（邪气清爽：吴语，意为非常清晰明白），否则容易吃亏上当。

    示例四（习语惯用语体）：
    上海话: 伊个人算盘打得响，嘴巴甜，就是心里向有点弯弯绕绕，跟伊打交道要留只眼睛。
    #mandarin: 他这个人精打细算（算盘打得响：吴语惯用语，比喻善于计算个人利益），嘴巴甜，就是心里有些弯弯绕绕（心思曲折，不坦率），跟他打交道要留个心眼。

    请严格按照以下格式输出普通话翻译结果：

    #mandarin: [你的普通话翻译结果]

    以下是需要翻译的上海话内容：
    「{segment_text}」
    """


def create_mandarin_to_english_prompt(segment_text: str) -> str:
    """
    Stage 2: Standard Mandarin (with Wu dialect glosses already embedded by
    Stage 1) → English.

    Simpler than the classical Chinese prompt: no wenyan unpacking needed.
    Parenthetical glosses from Stage 1 should be carried through into the
    English output to preserve dialect-specific meaning.
    """
    return f"""
    你是一位精通中文和英文的翻译专家。请将以下现代汉语内容翻译成流畅自然的英文。

    翻译要求：
    - **意译优先**：以传达原文含义为核心，避免机械直译；
    - 保留原文中括号内的解释性注释，将其自然融入英文译文中；
    - 保持原文的语气与情感色彩；
    - 不要添加额外评论或解释，只给出翻译结果；
    - 确保英文表达地道自然，避免在英文中夹杂中文词汇。

    【Few-Shot 示例】：

    示例一（口语对话体）：
    普通话: 我今天碰到隔壁的张老伯了，他说弄堂（上海石库门里弄小巷）里那家钱庄（旧上海私营传统银行）要关门了，你知道吗？
    #english: I ran into Old Mr. Zhang next door today — he said the money house (a traditional private bank from old Shanghai) in the lane (the narrow alley running through a shikumen residential block) is closing down. Did you hear about that?

    示例二（叙事体）：
    普通话: 话说很早以前，十六铺码头（旧上海最重要的客货两用码头）旁边有个跑单帮（独自从事小额贸易或走私的商人）的男子，专门替洋行（外国商行）里面传递重要文件，行动非常迅速，从来不出差错。
    #english: Long ago, beside the Shiliupu Wharf (once Shanghai's busiest passenger and cargo dock), there was a lone trader (an independent merchant who dealt in small goods, sometimes outside the law) who worked as a courier for foreign trading houses, delivering important documents with remarkable speed and never making a mistake.

    示例三（说明指令体）：
    普通话: 要做好这笔生意，第一要先摸清市场行情（行市：市场价格与供需状况），第二要找对渠道和关系网络，第三要把账目算得非常清楚，否则容易吃亏上当。
    #english: To handle this deal well: first, get a clear read on market conditions (prices, supply, and demand); second, identify the right channels and connections; third, keep your accounts meticulously clean — otherwise you will end up on the losing end.

    示例四（习语惯用语体）：
    普通话: 他这个人精打细算（算盘打得响：吴语惯用语，比喻善于计算个人利益），嘴巴甜，就是心里有些弯弯绕绕（心思曲折，不坦率），跟他打交道要留个心眼。
    #english: He is sharp with numbers and quick to calculate where his interests lie (literally "his abacus clicks loud" — a Wu dialect idiom for a calculating mind), smooth-tongued, but with an indirect way of thinking beneath the surface. You need to stay alert when dealing with him.

    请严格按照以下格式输出英文翻译结果：

    #english: [你的英文翻译结果]

    以下是需要翻译的普通话内容：
    「{segment_text}」
    """


def translate_shanghainese_segment(
    segment_text: str,
    model_type: str = "api",
    model_name: str = "deepseek-chat",
    max_tokens: int = 2000
) -> Optional[str]:
    """
    Two-stage translation for a single Shanghainese segment:
      Stage 1: Shanghainese → Mandarin  (Wu dialect glosses embedded)
      Stage 2: Mandarin → English
    """
    # Stage 1
    stage1_prompt = create_shanghainese_to_mandarin_prompt(segment_text)
    try:
        if model_type == "api":
            stage1_response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional Wu dialect and Mandarin translator."},
                    {"role": "user", "content": stage1_prompt}
                ],
            )
            stage1_raw = stage1_response.choices[0].message.content
        else:
            stage1_raw = get_gpt_pipeline(
                text=stage1_prompt,
                model_id=model_name,
                max_tokens=max_tokens,
                model_type=model_type
            )
    except Exception as e:
        print(f"Error in Shanghainese Stage 1 (→ Mandarin): {e}")
        return None

    mandarin_text = extract_mandarin_result(stage1_raw)
    if not mandarin_text:
        print("Warning: Stage 1 produced no #mandarin: output; falling back to raw Stage 1 text.")
        mandarin_text = stage1_raw or segment_text

    # Stage 2
    stage2_prompt = create_mandarin_to_english_prompt(mandarin_text)
    try:
        if model_type == "api":
            stage2_response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": stage2_prompt}
                ],
            )
            stage2_raw = stage2_response.choices[0].message.content
        else:
            stage2_raw = get_gpt_pipeline(
                text=stage2_prompt,
                model_id=model_name,
                max_tokens=max_tokens,
                model_type=model_type
            )
    except Exception as e:
        print(f"Error in Shanghainese Stage 2 (→ English): {e}")
        return None

    return extract_english_result(stage2_raw)


def serial_translate_shanghainese_segments(
    segments: List[str],
    model_type: str = "api",
    model_name: str = "deepseek-chat",
    max_tokens: int = 2000
) -> str:

    results = []

    for i, segment in enumerate(segments):
        print(f"Translating Shanghainese segment {i+1}/{len(segments)} ...")
        try:
            result = translate_shanghainese_segment(
                segment,
                model_type,
                model_name,
                max_tokens
            )

            if result is None:
                print(f"Warning: Shanghainese translation of segment {i+1} failed, using placeholder")
                result = f"[Translation of segment {i+1} failed]"

            results.append(result)
        except Exception as e:
            print(f"Error: Exception in Shanghainese segment {i+1}: {str(e)}")
            results.append(f"[Translation of segment {i+1} failed]")

    return " ".join(results)


def shanghainese_to_english(
    text: str,
    model_type: str = "api",
    model_name: str = "deepseek-chat",
    max_tokens: int = 2000
) -> Optional[str]:

    segments = preprocess_and_segment_text(text)

    return serial_translate_shanghainese_segments(
        segments,
        model_type,
        model_name,
        max_tokens
    )


# ---------------------------------------------------------------------------
# Cantonese pipeline  (two-stage: Cantonese → Mandarin → English)
# ---------------------------------------------------------------------------

def create_cantonese_to_mandarin_prompt(segment_text: str) -> str:
    """
    Stage 1: Cantonese (Yue dialect) → Standard Mandarin.

    Priority order:
      1. Convey the speaker's intended meaning faithfully (meaning > literal).
      2. Flag Yue-dialect-specific terms, particles, and idioms with a
         parenthetical gloss so meaning is not lost in Stage 2.
      3. Preserve register (conversational stays conversational, narrative
         stays narrative, etc.).

    Output tag: #mandarin:
    """
    return f"""
    你是一位精通粤语（广府话）和普通话的翻译专家。请将以下广东话内容翻译成标准普通话。

    翻译要求：
    - **意译优先**：以传达说话人的真实意图为核心，避免逐字对译；
    - 对于**粤语特有词汇、语气词或文化概念**，在括号内附上简短的解释性注释，例如：饮茶（粤港传统的茶楼品茗聚餐习俗）、骑楼（广州/香港常见的底层架空外廊式建筑）；
    - 保持原文的语气、语态和情感色彩（口语体保持口语，叙事体保持叙事）；
    - 不要添加评论或分析，只输出翻译结果；
    - 输出语言为流畅自然的标准普通话。

    【Few-Shot 示例】：

    示例一（口语对话体）：
    广东话: 喂，你今日去唔去铜锣湾行街啊？我听讲嗰边新开咗间茶楼，点心几好食㗎喎。
    #mandarin: 喂，你今天去不去铜锣湾（香港热闹的商业区）逛街啊？我听说那边新开了一家茶楼（粤港传统饮茶场所），点心（粤式小吃，通常在饮茶时以推车方式供应）很好吃呢。

    示例二（叙事体）：
    广东话: 话说旧时香港仔避风塘有个老渔民，专门喺夜里向西贡运货，手脚利落，从来唔出乱子。
    #mandarin: 话说旧时香港仔（香港南部一个渔港）避风塘（供渔船避风的人工港湾）有个老渔民，专门在夜里往西贡运货，动作麻利（手脚利落：粤语惯用语，指行动迅速敏捷），从来不出乱子。

    示例三（说明指令体）：
    广东话: 要做好呢单生意，其一要摸清行情，其二要搵对路子，其三要将账目算得清清楚楚，咪俾人坑咗先。
    #mandarin: 要做好这笔生意，第一要摸清市场行情，第二要找对渠道和关系网络，第三要将账目算得清清楚楚，否则容易被人坑害（坑：粤语，意为被欺骗或算计）。

    示例四（习语惯用语体）：
    广东话: 佢呢个人好识计，口甜舌滑，不过心里向有啲弯弯绕绕，同佢打交道要留只眼睛。
    #mandarin: 他这个人很善于算计（好识计：粤语，意为善于谋划盘算），嘴甜话巧（口甜舌滑：粤语成语，形容说话甜蜜圆滑），不过心里有些弯弯绕绕，与他打交道要留个心眼（留只眼睛：粤语，意为保持警惕）。

    请严格按照以下格式输出普通话翻译结果：

    #mandarin: [你的普通话翻译结果]

    以下是需要翻译的广东话内容：
    「{segment_text}」
    """


def translate_cantonese_segment(
    segment_text: str,
    model_type: str = "api",
    model_name: str = "deepseek-chat",
    max_tokens: int = 2000
) -> Optional[str]:
    """
    Two-stage translation for a single Cantonese segment:
      Stage 1: Cantonese → Mandarin  (Yue dialect glosses embedded)
      Stage 2: Mandarin → English
    """
    # Stage 1
    stage1_prompt = create_cantonese_to_mandarin_prompt(segment_text)
    try:
        if model_type == "api":
            stage1_response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional Cantonese and Mandarin translator."},
                    {"role": "user", "content": stage1_prompt}
                ],
            )
            stage1_raw = stage1_response.choices[0].message.content
        else:
            stage1_raw = get_gpt_pipeline(
                text=stage1_prompt,
                model_id=model_name,
                max_tokens=max_tokens,
                model_type=model_type
            )
    except Exception as e:
        print(f"Error in Cantonese Stage 1 (→ Mandarin): {e}")
        return None

    mandarin_text = extract_mandarin_result(stage1_raw)
    if not mandarin_text:
        print("Warning: Stage 1 produced no #mandarin: output; falling back to raw Stage 1 text.")
        mandarin_text = stage1_raw or segment_text

    # Stage 2 — reuse the dialect-agnostic Mandarin → English prompt
    stage2_prompt = create_mandarin_to_english_prompt(mandarin_text)
    try:
        if model_type == "api":
            stage2_response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": stage2_prompt}
                ],
            )
            stage2_raw = stage2_response.choices[0].message.content
        else:
            stage2_raw = get_gpt_pipeline(
                text=stage2_prompt,
                model_id=model_name,
                max_tokens=max_tokens,
                model_type=model_type
            )
    except Exception as e:
        print(f"Error in Cantonese Stage 2 (→ English): {e}")
        return None

    return extract_english_result(stage2_raw)


def serial_translate_cantonese_segments(
    segments: List[str],
    model_type: str = "api",
    model_name: str = "deepseek-chat",
    max_tokens: int = 2000
) -> str:

    results = []

    for i, segment in enumerate(segments):
        print(f"Translating Cantonese segment {i+1}/{len(segments)} ...")
        try:
            result = translate_cantonese_segment(
                segment,
                model_type,
                model_name,
                max_tokens
            )

            if result is None:
                print(f"Warning: Cantonese translation of segment {i+1} failed, using placeholder")
                result = f"[Translation of segment {i+1} failed]"

            results.append(result)
        except Exception as e:
            print(f"Error: Exception in Cantonese segment {i+1}: {str(e)}")
            results.append(f"[Translation of segment {i+1} failed]")

    return " ".join(results)


def cantonese_to_english(
    text: str,
    model_type: str = "api",
    model_name: str = "deepseek-chat",
    max_tokens: int = 2000
) -> Optional[str]:

    segments = preprocess_and_segment_text(text)

    return serial_translate_cantonese_segments(
        segments,
        model_type,
        model_name,
        max_tokens
    )


# ---------------------------------------------------------------------------
# Dialect dispatcher
# ---------------------------------------------------------------------------

_SUPPORTED_DIALECTS = {"classical_chinese", "shanghainese", "cantonese", "generic"}


def translate_to_english(
    text: str,
    dialect: str = "classical_chinese",
    model_type: str = "api",
    model_name: str = "deepseek-chat",
    max_tokens: int = 2000
) -> Optional[str]:
    """
    Top-level entry point for dialect-to-English translation.

    dialect: one of "classical_chinese" | "shanghainese" | "cantonese" | "generic"

    classical_chinese_to_english() remains callable directly for
    backwards compatibility with existing gen.py call sites. The "generic"
    control dialect routes through the same single-stage Chinese→English path
    (no Wu/Yue gloss step needed, since target responses are plain Mandarin
    or Mandarin-register text).
    """
    if dialect not in _SUPPORTED_DIALECTS:
        raise ValueError(
            f"Unsupported dialect '{dialect}'. "
            f"Supported dialects: {sorted(_SUPPORTED_DIALECTS)}"
        )

    if dialect == "shanghainese":
        return shanghainese_to_english(text, model_type, model_name, max_tokens)

    if dialect == "cantonese":
        return cantonese_to_english(text, model_type, model_name, max_tokens)

    # classical_chinese and generic both use the single-stage Chinese→English path.
    return classical_chinese_to_english(text, model_type, model_name, max_tokens)
