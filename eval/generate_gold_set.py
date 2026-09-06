import json
from pathlib import Path

from faker import Faker

OUTPUT = Path("data/synthetic/gold_set.jsonl")
SEED = 42
N_DOCS = 30

Segment = tuple[str, str | None]

# Hard cases planted deliberately: particles/hyphens, surnames that are also
# common nouns, and non-French-origin names — see claude.md for why.
NAME_ORIGIN: dict[str, str] = {
    "Gérard-Frédéric Joly": "french",
    "Alexandre Traore": "french",
    "Marie de la Rochefoucauld": "french_particle",
    "Jean Dupont-Moretti": "french_hyphenated",
    "Fleury Rose": "french_common_noun_surname",
    "Minh Nguyen": "non_french",
    "Awa Diallo": "non_french",
    "Piotr Kowalski": "non_french",
    "Yasmine Ben Salah": "non_french",
}


def assemble(segments: list[Segment]) -> tuple[str, list[dict]]:
    """Concatenate segments"""
    parts: list[str] = []
    entities: list[dict] = []
    offset = 0
    for text, entity_type in segments:
        if entity_type is not None:
            entities.append(
                {
                    "type": entity_type,
                    "start": offset,
                    "end": offset + len(text),
                    "value": text,
                }
            )
        parts.append(text)
        offset += len(text)
    return "".join(parts), entities


def hr_letter(fake: Faker, doc_id: str) -> dict:
    manager, employee = fake.random_elements(
        elements=list(NAME_ORIGIN), length=2, unique=True
    )
    address = fake.address().replace("\n", ", ")  # flatten
    birth_date = fake.date_of_birth(minimum_age=22, maximum_age=77).strftime("%d/%m/%Y")
    hire_date = fake.date_between(start_date="-8y").strftime("%d/%m/%Y")
    salary = f"{fake.random_int(1800, 7000)}€"

    segments: list[Segment] = [
        (fake.company(), None),  # org names are out of scope, see claude.md
        ("\nObjet : attestation d'emploi\n\n", None),
        ("Je soussigné(e) ", None),
        (manager, "PERSON"),
        (", responsable des ressources humaines, atteste que ", None),
        (employee, "PERSON"),
        (", né(e) le ", None),
        (birth_date, "DATE_TIME"),
        (", demeurant ", None),
        (address, "LOCATION"),
        (", est employé(e) dans notre société depuis le ", None),
        (hire_date, "DATE_TIME"),
        (".\n\n", None),
        ("Numéro de sécurité sociale : ", None),
        (fake.ssn(), "FR_NIR"),  # NIR control key not verified by Faker
        ("\nRémunération mensuelle brute : ", None),
        (salary, "SALARY"),
        ("\nVersée sur le compte ", None),
        (fake.iban(), "IBAN_CODE"),
        (".\n\n", None),
        # Deliberate precision trap: a business amount that must NOT be masked.
        (
            "Attestation délivrée dans le cadre du marché n° 2024-118, d'un montant de 145 000 €.\n\n",
            None,
        ),
        ("Contact : ", None),
        (fake.email(), "EMAIL_ADDRESS"),
        (" — ", None),
        (fake.phone_number(), "PHONE_NUMBER"),
        ("\nSIRET : ", None),
        (fake.siret(), "FR_SIRET"),
        ("\n", None),
    ]

    text, entities = assemble(segments)
    for entity in entities:
        if entity["type"] == "PERSON":
            entity["name_origin"] = NAME_ORIGIN[entity["value"]]

    return {
        "doc_id": doc_id,
        "text": text,
        "entities": entities,
        "meta": {"template": "hr_letter", "seed": SEED},
    }


def main() -> None:
    fake = Faker("fr_FR")
    Faker.seed(SEED)

    docs = [hr_letter(fake, f"hr_letter_{i:03d}") for i in range(N_DOCS)]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    n_entities = sum(len(d["entities"]) for d in docs)
    print(f"{len(docs)} documents, {n_entities} entities -> {OUTPUT}")


if __name__ == "__main__":
    main()
