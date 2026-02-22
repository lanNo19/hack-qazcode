"""
parse_icd_codes.py

Builds a dictionary mapping ICD-10 codes (with dots) to their text descriptions.
Uses both precise (A00.9) and parental (A02) codes to strictly divide text chunks.
Extracts codes wrapped in parentheses and converts cyrillic code letters to latin.
"""

import json
import re
import argparse

# 1. Универсальная регулярка для МКБ кодов. 
# Ловит и точные (A00.9), и родительские (A02).
# Ловит буквы латиницы и кириллицы.
# Игнорирует скобки вокруг кода благодаря \)? и \(? вне захватывающей группы.
ICD_ALL_RE = re.compile(r'\(?([A-ZА-ЯЁ]\d{2}(?:\.\d{1,2})?)\)?')

# 2. Регулярка для поиска слова с заглавной буквы (пропускает ALL-CAPS аббревиатуры)
# Ищет пробел, затем Заглавную, затем минимум одну строчную.
CAPITAL_WORD_RE = re.compile(r'\s+[A-ZА-ЯЁ][a-zа-яё]+')

# Таблица транслитерации кириллических букв МКБ-10 в латинские 
# (для защиты от омоглифов и случайного дублирования ключей)
CYR_TO_LAT = str.maketrans("АВЕКМНОРСТХУ", "ABEKMHOPCTXY")


def clean_description(raw: str) -> str:
    """Очищает мусор в начале и конце описания."""
    # Убираем ведущие тире, плюсы, двоеточия, закрывающиеся скобки и пробелы
    desc = re.sub(r'^[\s\u2013\u2014\-\:\)\+]+', '', raw)
    
    # Схлопываем лишние пробелы внутри
    desc = re.sub(r'\s+', ' ', desc).strip()
    
    # Убираем "висящие" номера в конце, если они остались (например, " 1.2")
    desc = re.sub(r'[\s\d\.]+$', '', desc).strip()
    
    return desc


def parse_icd_descriptions(text: str) -> dict:
    result = {}
    matches = list(ICD_ALL_RE.finditer(text))
    
    for i, match in enumerate(matches):
        code = match.group(1)
        
        # Начало описания — сразу после кода (и его возможной скобки)
        desc_start = match.end()
        
        if i + 1 < len(matches):
            # Если есть следующий код (даже родительский), режем строго до него
            desc_end = matches[i+1].start()
            raw_desc = text[desc_start:desc_end]
        else:
            # ЛОГИКА ПОСЛЕДНЕГО КОДА
            tail = text[desc_start:]
            
            # Сначала убираем мусор в начале, чтобы первое слово диагноза стояло с 0 индекса
            tail = re.sub(r'^[\s\u2013\u2014\-\:\)\+]+', '', tail)
            
            # Ищем следующее слово с заглавной буквы (но не ALL-CAPS).
            # Из-за \s+ в регулярке, оно автоматически пропустит первое слово диагноза.
            cap_match = CAPITAL_WORD_RE.search(tail)
            
            if cap_match:
                # Отрезаем всё, начиная с найденного слова (например, " Дата")
                raw_desc = tail[:cap_match.start()]
            else:
                raw_desc = tail
                
        # СОХРАНЯЕМ ТОЛЬКО ТОЧНЫЕ КОДЫ (с точкой)
        if '.' in code:
            # Защита от омоглифов: А02.0 (кириллица) -> A02.0 (латиница)
            normalized_code = code.translate(CYR_TO_LAT)
            
            desc = clean_description(raw_desc)
            
            # Сохраняем, если описание осмысленное
            if desc and len(desc) > 3:
                # ЛОГИКА: Сохраняем наиболее КОРОТКОЕ описание для данного кода
                if normalized_code not in result or len(desc) < len(result[normalized_code]):
                    result[normalized_code] = desc
                    
    return result


def build_icd_dict(corpus_path: str) -> dict:
    combined = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            
            # Ищем коды только в первой части документа (в Введении)
            header_text = text[:3000]
            
            descriptions = parse_icd_descriptions(header_text)
            
            for code, desc in descriptions.items():
                if code not in combined:
                    combined[code] = desc
                elif len(desc) < len(combined[code]):
                    # Сохраняем наикратчайший вариант по всему корпусу
                    combined[code] = desc

    return combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build clean ICD-10 dictionary')
    parser.add_argument('--corpus', default='corpus/protocols_corpus.jsonl')
    parser.add_argument('--output', default='icd_descriptions.json')
    args = parser.parse_args()

    print(f"Reading corpus: {args.corpus}")
    icd_dict = build_icd_dict(args.corpus)
    
    # Сортируем по ключам для удобного чтения JSON
    icd_dict_sorted = dict(sorted(icd_dict.items()))

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(icd_dict_sorted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(icd_dict_sorted)} clean codes to {args.output}")