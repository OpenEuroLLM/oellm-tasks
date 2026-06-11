from functools import partial

import datasets


##DATA LOADING
def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        # Extract nested fields and flatten them for easy templating
        return {
            "context": doc["input"]["context"],
            "question": doc["input"]["question"],
            "answer": doc["input"]["answer"],
            "rubric_fc": doc["input"]["rubrics"].get("FC", ""),
            "rubric_pc": doc["input"]["rubrics"].get("PC", ""),
            "rubric_nc": doc["input"]["rubrics"].get("NC", ""),
            # Ensure the label is an integer so it maps correctly to the index of our choices
            "label": int(doc["label"])
        }

    return dataset.map(_process_doc)


def process_docs_yes_no(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc, checked_label):
        # Extract nested fields and flatten them for easy templating
        return {
            "context": doc["input"]["context"],
            "question": doc["input"]["question"],
            "answer": doc["input"]["answer"],
            "rubric_fc": doc["input"]["rubrics"].get("FC", ""),
            "rubric_pc": doc["input"]["rubrics"].get("PC", ""),
            "rubric_nc": doc["input"]["rubrics"].get("NC", ""),
            # Ensure the label is an integer so it maps correctly to the index of our choices
            "check_string": f"Can this answer be rated with {checked_label}.",
            "label": int(1 if doc["label"] == checked_label else 0)
        }

    return datasets.concatenate_datasets([dataset.map(partial(_process_doc, checked_label=0)),
                                          dataset.map(partial(_process_doc, checked_label=1)),
                                          dataset.map(partial(_process_doc, checked_label=2))])


def process_docs_lang_domain(dataset, lang, domain):
    return process_docs(dataset.filter(lambda x: x["lang"] == lang and x["domain"] == domain))


def process_docs_lang(dataset, lang):
    return process_docs(dataset.filter(lambda x: x["lang"] == lang))


pairs = [('cs', 'OECD_public'), ('cs', 'demagog'), ('cs', 'popular_science'), ('da', 'OECD_public'), ('da', 'demagog'),
         ('da', 'popular_science'),
         ('de', 'OECD_public'), ('de', 'demagog'), ('de', 'popular_science'), ('el', 'OECD_public'), ('el', 'demagog'),
         ('el', 'popular_science'),
         ('en', 'OECD_public'), ('en', 'demagog'), ('en', 'popular_science'), ('fi', 'OECD_public'), ('fi', 'demagog'),
         ('fi', 'popular_science'),
         ('ga', 'OECD_public'), ('ga', 'demagog'), ('ga', 'popular_science'), ('hu', 'OECD_public'), ('hu', 'demagog'),
         ('hu', 'popular_science'),
         ('pt', 'OECD_public'), ('pt', 'demagog'), ('pt', 'popular_science'), ('ro', 'OECD_public'), ('ro', 'demagog'),
         ('ro', 'popular_science'),
         ('sr', 'OECD_public'), ('sr', 'demagog'), ('sr', 'popular_science'), ('sv', 'OECD_public'), ('sv', 'demagog'),
         ('sv', 'popular_science'),
         ('uk', 'OECD_public'), ('uk', 'demagog'), ('uk', 'popular_science')]

for lang, domain in pairs:
    exec(f"""process_{lang} = partial(process_docs_lang, lang=lang)""")
    exec(f"""process_{lang}_{domain} = partial(process_docs_lang_domain, lang=lang, domain=domain)""")


##DATA MANAGEMENT
def doc_to_text_hybrid(doc) -> str:
    # Extract nested fields directly from the original document structure
    context = doc["input"]["context"]
    question = doc["input"]["question"]
    answer = doc["input"]["answer"]

    # Extract rubrics
    rubric_nc = doc["input"]["rubrics"].get("NC", "").split(splits)[0]
    rubric_pc = doc["input"]["rubrics"].get("PC", "").split(splits)[0]
    rubric_fc = doc["input"]["rubrics"].get("FC", "").split(splits)[0]

    # Construct the prompt using full words for the credit levels
    prompt = (
        "You are an expert grader evaluating a student's answer based on a specific rubric.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Student Answer:\n{answer}\n\n"
        "Rubric:\n"
        f"0 (No Credit):\n{rubric_nc}\n\n"
        f"1 (Partial Credit):\n{rubric_pc}\n\n"
        f"2 (Full Credit):\n{rubric_fc}\n\n"
        "Based on the rubric, assign the appropriate score (0, 1, or 2) to the student's answer.\nScore:"
    )
    return prompt

splits = "Examples:\n"
def doc_to_text_long(doc) -> str:
    # Extract nested fields directly from the original document structure
    context = doc["input"]["context"]
    question = doc["input"]["question"]
    answer = doc["input"]["answer"]

    # Extract rubrics
    rubric_nc = doc["input"]["rubrics"].get("NC", "").split(splits)[0]
    rubric_pc = doc["input"]["rubrics"].get("PC", "").split(splits)[0]
    rubric_fc = doc["input"]["rubrics"].get("FC", "").split(splits)[0]

    # Construct the prompt using full words for the credit levels
    prompt = (
        "You are an expert grader evaluating a student's answer based on a specific rubric.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Student Answer:\n{answer}\n\n"
        "Rubric:\n"
        f"No Credit:\n{rubric_nc}\n\n"
        f"Partial Credit:\n{rubric_pc}\n\n"
        f"Full Credit:\n{rubric_fc}\n\n"
        "Based on the rubric, assign the appropriate score (0, 1, or 2) to the student's answer.\nScore:"
    )
    return prompt

def doc_to_text_numbers(doc) -> str:
    # Extract nested fields directly from the original document structure
    context = doc["input"]["context"]
    question = doc["input"]["question"]
    answer = doc["input"]["answer"]

    # Extract rubrics
    rubric_nc = doc["input"]["rubrics"].get("NC", "").split(splits)[0]
    rubric_pc = doc["input"]["rubrics"].get("PC", "").split(splits)[0]
    rubric_fc = doc["input"]["rubrics"].get("FC", "").split(splits)[0]

    # Construct the prompt using full words for the credit levels
    prompt = (
        "You are an expert grader evaluating a student's answer based on a specific rubric.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Student Answer:\n{answer}\n\n"
        "Rubric:\n"
        f"0:\n{rubric_nc}\n\n"
        f"1:\n{rubric_pc}\n\n"
        f"2:\n{rubric_fc}\n\n"
        "Based on the rubric, assign the appropriate score (0, 1, or 2) to the student's answer.\nScore:"
    )
    return prompt

def list_fewshot_samples() -> list[dict]:
    return [
        # Example 1: No Credit (Score: 0)
        {
            "context": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
            "question": "What three specific inputs do plants need for photosynthesis?",
            "answer": "Plants need soil, nutrients, and fertilizer to grow tall.",
            "rubric_nc": "Fails to mention any of the three required inputs (sunlight, water, carbon dioxide).",
            "rubric_pc": "Mentions 1 or 2 of the correct inputs.",
            "rubric_fc": "Correctly identifies all three: sunlight, water, and carbon dioxide.",
            "label": 0
        },

        # Example 2: Partial Credit (Score: 1)
        {
            "context": "The Apollo 11 mission landed on the Moon on July 20, 1969. The historic crew consisted of Neil Armstrong, Buzz Aldrin, and Michael Collins.",
            "question": "Who were the astronauts on the Apollo 11 mission, and what year did they land?",
            "answer": "The mission happened in 1969 and the astronauts were Neil Armstrong and Buzz Aldrin.",
            "rubric_nc": "Provides an incorrect year and incorrect/missing names.",
            "rubric_pc": "Provides the correct year but misses at least one astronaut, OR names all three astronauts but misses/gets the year wrong.",
            "rubric_fc": "Correctly names all three astronauts (Armstrong, Aldrin, Collins) AND the year 1969.",
            "label": 1
        },

        # Example 3: Full Credit (Score: 2)
        {
            "context": "Under standard atmospheric pressure, water freezes at 0°C (32°F) and boils at 100°C (212°F).",
            "question": "At what temperatures does water freeze and boil in Celsius?",
            "answer": "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
            "rubric_nc": "Provides incorrect temperatures for both, or answers in Fahrenheit.",
            "rubric_pc": "Correctly states either the freezing point or the boiling point in Celsius, but not both.",
            "rubric_fc": "Accurately states both 0°C for freezing and 100°C for boiling.",
            "label": 2
        }
    ]

