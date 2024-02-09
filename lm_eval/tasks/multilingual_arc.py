"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf

The ARC dataset consists of 7,787 science exam questions drawn from a variety
of sources, including science questions provided under license by a research
partner affiliated with AI2. These are text-only, English language exam questions
that span several grade levels as indicated in the files. Each question has a
multiple choice structure (typically 4 answer options). The questions are sorted
into a Challenge Set of 2,590 “hard” questions (those that both a retrieval and
a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

Homepage: https://allenai.org/data/arc
"""
from lm_eval.base import MultipleChoiceTask

_CITATION = """
@article{Clark2018ThinkYH,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal={ArXiv},
  year={2018},
  volume={abs/1803.05457}
}
"""

LANGS = 'ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh'.split(',')
LANGS = ["ro"]
FS_VALUES = [0,1,3,5,10,25]

def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {arc_vi: Task, arc_bn: Task}
    """
    return {f"arc_{lang}_fs{fs}_{prompt}": create_task(lang, fs, prompt) for fs in FS_VALUES for lang in LANGS for prompt in ["foundational", "chat"]}


def create_task(lang, fs, model_type):

    class ATest(MultilingualARC):
        def __init__(self):
            super().__init__(lang, fs=fs, model_type=model_type)

    return ATest


class MultilingualARC(MultipleChoiceTask):

    def __init__(self, lang, fs, model_type, **kwargs):
        self.VERSION = 0
        self.lang = lang
        self.DATASET_NAME = f"arc_{lang}"
        self.DATASET_PATH = 'datasets/m_arc'
        self.NUM_FEW_SHOT = fs
        self.model_type = model_type
        print("ARC FEWSHOT:", self.NUM_FEW_SHOT)
        print("Model type:", model_type)
        super().__init__(**kwargs)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # NOTE:
        # print("doc:", doc)
        if self.model_type == "foundational":
            bos = "Întrebare: "
            eos = "\nRăspuns:"
        elif self.model_type == "chat":
            bos = ""
            eos = ""

        out_doc = {
            "id": doc["id"],
            "query": bos + doc["question"] + eos,
            "choices": doc["choices"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }

        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
