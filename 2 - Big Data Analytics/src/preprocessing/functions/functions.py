import re
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import spacy
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, FloatType

__SERVER = 'http://localhost:2222/rest'


@udf(StringType())
def generate_title(title):
    return title.title()


@udf(StringType())
def generate_career(career):
    return str(career) if career else ''


@udf(StringType())
def generate_code(acronym, catalog):
    return f'{acronym}{catalog}'


@udf(FloatType())
def generate_credit(class_unit):
    return float(class_unit) if class_unit else 0.0


@udf(ArrayType(StringType()))
def generate_requisites(prerequisite, cross_listed):
    prerequisite = prerequisite if prerequisite else ''
    cross_listed = cross_listed if cross_listed else ''
    text = f'{prerequisite} {cross_listed}'
    values = set()
    for match in re.finditer(r'[A-Z]{3,4} ?\d{3,4}', text):
        values.add(re.sub(' ', '', match.group()))

    return list(values)


@udf(StringType())
def generate_description(description):
    rules = [('', ['[?]',
                   '[\r]',
                   '[\\]',
                   '[\\"]',
                   '<[^>]*>',
                   '`',
                   '[Pp]?(lease)? [Ss]ee [a-zA-Z]+ [Cc]alendar[.]?',
                   '[Nn][Oo][Tt][Ee]( [a-zA-Z0-9]{0,2})?:[^.]*[.]?',
                   'Note that[^.]*[.]?',
                   'Prerequisite[^:]*:[^.]*[.]?',
                   '[^.]*prereq[^.]*[.]?',
                   '~',
                   '[*][^.*]*[*.]?',
                   '\[BANNER[^\]]*[\]]',
                   '[^.]*may not take this course[^.]*[.]?',
                   '[^.]*received credit[^.]*[.]?',
                   'Lectures?:[^.]*[.]?',
                   'Tutorials?[^.]*[.]?',
                   'Laboratory[^.]*[.]?',
                   ]),
             ('/', ['[\\\/]']),
             (' ', ['[\n]+',
                    '[\t]+']),
             (' ', [' {2,}'])
             ]

    for substitute, patterns in rules:
        pattern = '|'.join(f'({x})' for x in patterns)
        description = re.sub(pattern, substitute, description)
    return description


def _init_nlp():
    nlp = spacy.blank('en')
    nlp.add_pipe('dbpedia_spotlight',
                 config={'dbpedia_rest_endpoint': __SERVER, 'confidence': 0.35})
    return nlp


def _generate_concepts(description, nlp):
    concepts = set()
    if description:
        try:
            for ent in nlp(description).ents:
                concepts.add(str(ent.kb_id_))
        except Exception:
            pass
    return list(concepts)


def generate_pandas_concepts(df: pd.DataFrame) -> pd.DataFrame:
    nlp = _init_nlp()
    concepts = {'ID': [], 'concepts': []}

    with ThreadPoolExecutor(max_workers=5) as executor:
        flist = {executor.submit(_generate_concepts, row['description'], nlp): row['ID'] for _, row in df.iterrows()}
        for future in futures.as_completed(flist):
            course_id = flist[future]
            try:
                concepts['ID'].append(course_id)
                concepts['concepts'].append(future.result())
            except Exception:
                pass

    concepts = pd.DataFrame.from_dict(concepts)
    return pd.merge(df, concepts, on=['ID'])


@udf(ArrayType(StringType()))
def generate_concepts(description):
    nlp = _init_nlp()
    return _generate_concepts(description, nlp)
