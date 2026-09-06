import re

from presidio_analyzer import Pattern, PatternRecognizer


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _fr_nir_ok(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 15:
        return False
    key = 97 - (int(digits[:13]) % 97)
    return key == int(digits[13:])


def _fr_siret_ok(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return len(digits) == 14 and _luhn_ok(digits)


class FrNirRecognizer(PatternRecognizer):
    PATTERNS = [Pattern(name="fr_nir", regex=r"\b[12]\d{14}\b", score=0.3)]

    def __init__(self):
        super().__init__(
            supported_entity="FR_NIR",
            patterns=self.PATTERNS,
            supported_language="fr",
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        return True if _fr_nir_ok(pattern_text) else False


class FrSiretRecognizer(PatternRecognizer):
    PATTERNS = [Pattern(name="fr_siret", regex=r"\b\d{3} ?\d{3} ?\d{3} ?\d{5}\b", score=0.3)]

    def __init__(self):
        super().__init__(
            supported_entity="FR_SIRET",
            patterns=self.PATTERNS,
            supported_language="fr",
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        return True if _fr_siret_ok(pattern_text) else False


class SalaryRecognizer(PatternRecognizer):
    PATTERNS = [Pattern(name="salary_amount", regex=r"\b\d+(?:[ .]\d{3})*(?:,\d{2})?\s?€", score=0.2)]
    CONTEXT = ["rémunération", "salaire", "brut", "net", "paie", "mensuelle"]

    def __init__(self):
        super().__init__(
            supported_entity="SALARY",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="fr",
        )
