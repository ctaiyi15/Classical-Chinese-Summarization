import time
from googletrans import Translator

async def translate (source_path, src_lang, dest_lang):
    start_time = time.perf_counter()
    translator = Translator()
    translated_lines = []
    with open(source_path,'r', encoding='utf-8') as file:
        lines = file.readlines()
        count = len(lines)
        for line in lines:
            result = await translator.translate(line, src=src_lang, dest=dest_lang)
            translated_lines.append(result.text + '\n')
    end_time = time.perf_counter()
    return end_time-start_time, count, translated_lines

async def one_way_translate(input_root, output_root, source_lang, target_lang):
    start_time  = time.perf_counter()
    for file_path in input_root.rglob("target.txt"):
        relative_path = file_path.relative_to(input_root)

        output_file = output_root / relative_path
        if output_file.exists():
            continue

        try:
            time_elapsed, line_count, translated_lines = await translate(file_path, source_lang, target_lang)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(''.join(translated_lines), encoding="utf-8")
            print(f"Finished: {relative_path} for {time_elapsed:.2f}s ({line_count} lines)")
        except Exception as e:
            print(f'FAILED on {relative_path} due to {e}')

    print(f'Finished translating all file in {time.perf_counter()-start_time:.2f}s')