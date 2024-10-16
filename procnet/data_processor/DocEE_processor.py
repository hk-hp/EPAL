import logging
from typing import List, Dict
from procnet.data_processor.basic_processor import BasicProcessor
from procnet.conf.global_config_manager import GlobalConfigManager
from procnet.data_example.DocEEexample import DocEEDocumentExample, DocEEEntity, DocEELabel
from procnet.utils.util_data import UtilData
import json


class DocEEProcessor(BasicProcessor):
    def __init__(self, data_name):
        super().__init__()
        self.data_path = GlobalConfigManager.get_dataset_path()
        logging.debug("Path: {}".format(self.data_path))

        self.train_path = self.data_path / data_name / "train.json"
        self.dev_path = self.data_path / data_name / "dev.json"
        self.test_path = self.data_path / data_name / "test.json"
        logging.debug("train_path: {}".format(self.train_path))
        logging.debug("dev_path: {}".format(self.dev_path))
        logging.debug("test_path: {}".format(self.test_path))

        if data_name == 'ChFinAnn':
            self.train_json = UtilData.read_raw_json_file(self.train_path)
            self.dev_json = UtilData.read_raw_json_file(self.dev_path)[:1]
            self.test_json = UtilData.read_raw_json_file(self.test_path)

            self.train_docs: List[DocEEDocumentExample] = self.parse_json_all(self.train_json)[:100]
            self.dev_docs: List[DocEEDocumentExample] = self.parse_json_all(self.dev_json)[:1]
            self.test_docs: List[DocEEDocumentExample] = self.parse_json_all(self.test_json)[:1000]

            self.SCHEMA = DocEELabel.EVENT_SCHEMA
            self.SCHEMA_KEY_ENG_CHN = DocEELabel.KEY_ENG_CHN
            self.SCHEMA_KEY_CHN_ENG = DocEELabel.KEY_CHN_ENG
        elif data_name == 'DuEE_Fin':
            self.train_json = UtilData.read_raw_json_file(self.train_path)
            self.test_json = UtilData.read_raw_json_file(self.test_path)

            self.train_docs: List[DocEEDocumentExample] = self.parse_json_all(self.train_json)
            self.dev_docs: List[DocEEDocumentExample] = self.parse_json_all(self.test_json)[:1]
            self.test_docs: List[DocEEDocumentExample] = self.parse_json_all(self.test_json)

            with open('Data/DuEE_Fin/duee_fin_event_schema.json', 'r', encoding='utf-8') as file:
                self.SCHEMA = {}
                for line in file.readlines():
                    one = json.loads(line)
                    self.SCHEMA[one['event_type']] = []
                    for item in one['role_list']:
                        self.SCHEMA[one['event_type']].append(item['role'])
        
            self.SCHEMA_KEY_ENG_CHN = None
            self.SCHEMA_KEY_CHN_ENG = None
        else:
            exit(0)
        
    def parse_json_all(self, json) -> List[DocEEDocumentExample]:
        docs = []
        for one in json:
            doc_id: str = one[0]
            data = one[1]
            sentences: List[str] = data['sentences']
            ann_mspan2dranges: Dict[str, List[list]] = data['ann_mspan2dranges']
            ann_mspan2guess_field: Dict[str, str] = data['ann_mspan2guess_field']
            recguid_eventname_eventdict_list = data['recguid_eventname_eventdict_list']

            assert len(ann_mspan2dranges) == len(ann_mspan2guess_field)
            entities = []
            for k, v in ann_mspan2dranges.items():
                entity = DocEEEntity(span=k, positions=v, field=ann_mspan2guess_field[k])
                entities.append(entity)

            events = []
            for x in recguid_eventname_eventdict_list:
                event = {'EventType': x[1]}
                for k, v in x[2].items():
                    event[k] = v
                events.append(event)

            doc = DocEEDocumentExample(doc_id=doc_id, sentences=sentences, entities=entities, events=events)

            for entity in doc.entities:
                for pos in entity.positions:
                    assert entity.span == doc.sentences[pos[0]][pos[1]:pos[2]]
            
            docs.append(doc)
        return docs

