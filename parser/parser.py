from web.parser.habr.habr import start_habr_parser
from web.parser.fl_ru.parser import start_fl_parser
def start_parser():
    start_habr_parser()
    start_fl_parser()


if __name__ == "__main__":
    start_parser()
