from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Model location
MODEL_PATH = ".data"

# Language translation model name
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

# Language translation model supported by the model
SUPPORTED_LANGUAGES = {
    "af": "af_ZA",
    "ar": "ar_AR",
    "az": "az_AZ",
    "bn": "bn_IN",
    "cs": "cs_CZ",
    "de": "de_DE",
    "en": "en_XX",
    "es": "es_XX",
    "et": "et_EE",
    "fa": "fa_IR",
    "fi": "fi_FI",
    "fr": "fr_XX",
    "gl": "gl_ES",
    "gu": "gu_IN",
    "he": "he_IL",
    "hi": "hi_IN",
    "hr": "hr_HR",
    "id": "id_ID",
    "it": "it_IT",
    "ja": "ja_XX",
    "ka": "ka_GE",
    "kk": "kk_KZ",
    "km": "km_KH",
    "ko": "ko_KR",
    "lt": "lt_LT",
    "lv": "lv_LV",
    "mk": "mk_MK",
    "ml": "ml_IN",
    "mn": "mn_MN",
    "mr": "mr_IN",
    "my": "my_MM",
    "ne": "ne_NP",
    "nl": "nl_XX",
    "pl": "pl_PL",
    "ps": "ps_AF",
    "pt": "pt_XX",
    "ro": "ro_RO",
    "ru": "ru_RU",
    "si": "si_LK",
    "sl": "sl_SI",
    "sv": "sv_SE",
    "sw": "sw_KE",
    "ta": "ta_IN",
    "te": "te_IN",
    "th": "th_TH",
    "tl": "tl_XX",
    "tr": "tr_TR",
    "uk": "uk_UA",
    "ur": "ur_PK",
    "vi": "vi_VN",
    "xh": "xh_ZA",
    "zh": "zh_CN",
}


class MachineTranslation():
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.model = MBartForConditionalGeneration.from_pretrained(
            MODEL_NAME, cache_dir=MODEL_PATH)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            MODEL_NAME, cache_dir=MODEL_PATH)
        return self

    def translate(self, text, src_lang, target_lang) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")

        if src_lang not in SUPPORTED_LANGUAGES or target_lang not in SUPPORTED_LANGUAGES:
            raise ValueError("Unsupported language")

        src_lang = SUPPORTED_LANGUAGES[src_lang]
        target_lang = SUPPORTED_LANGUAGES[target_lang]

        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang])
        translation = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)

        return translation[0]


if __name__ == "__main__":
    text = "Hello, how are you?"
    src_lang, target_lang = "en", "fa"
    model = MachineTranslation().load_model()
    translation = model.translate(text, src_lang, target_lang)
    print(translation)
