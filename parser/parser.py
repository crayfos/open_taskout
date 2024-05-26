from web.parser.habr.habr import start_habr_parser
from web.parser.fl_ru.parser import start_fl_parser
from web.parser.kwork.kwork_parser import start_kwork_parser
from web.parser.youdo.youdo_parser import start_youdo_parser

def start_parser():
    start_habr_parser()
    start_fl_parser()
    start_kwork_parser()
    start_youdo_parser()


if __name__ == "__main__":
    start_parser()
