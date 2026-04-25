from translation import one_way_translate
from pathlib import Path
import asyncio

input_root = Path("../data/raw/后汉书")
output_root = Path("../data/processed/translated/后汉书")
output2_root = Path("../data/processed/translated_back/后汉书")


if __name__ == "__main__":
    asyncio.run(
        one_way_translate(
            input_root,
            output_root,
            "zh-cn",
            "en"
        )
    )
    asyncio.run(
        one_way_translate(
            output_root,
            output2_root,
            "en",
            "zh-cn"
        )
    )