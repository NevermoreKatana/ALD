import click
import logging

from main import main_train, main_detect_one_day
from config import Settings


settings = Settings()


@click.command()
@click.option('--train', default=False, help='Start train new model')
@click.option('--test', default=False, help='Start test model')
@click.option('--html', default=False, help='Generate HTML report')
def model_pipeline(train: bool, test:bool, html:bool):
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    
    BASE_LOG_DIR = settings.BASE_LOG_DIR
    MODEL_PATH = settings.MODEL_PATH
    ONE_DAY_LOG_PATH = settings.ONE_DAY_LOG_PATH
    
    if train:
        trained_model = main_train(settings.BASE_LOG_DIR, model_path=settings.MODEL_PATH)
    
    if test is True and html is True:
        main_detect_one_day(settings.MODEL_PATH, settings.ONE_DAY_LOG_PATH, html_report=True)
    else:
        main_detect_one_day(settings.MODEL_PATH, settings.ONE_DAY_LOG_PATH, html_report=False)


if __name__ == '__main__':
    model_pipeline()
