import sys
import os


def start_parser():
    with open("habr/habr.py", encoding="utf-8") as file:
        code = file.read()
        exec(code)


if __name__ == "__main__":
    start_parser()
