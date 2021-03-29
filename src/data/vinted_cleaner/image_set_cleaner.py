import sys
from src.data.vinted_cleaner.gui_image_selector import MainWindow
import argparse
from PyQt5.QtWidgets import QApplication
from src.data.vinted_cleaner.file_processing import *



def verify_input(FLAGS):
    """
    This method will check the values given by the user.
    :param _: Parser
    :return: Nothing
    """

    if not os.path.exists(FLAGS.directory_dir):
        raise AssertionError('Image directory not found.')


def main(FLAGS):

    verify_input(FLAGS)
    app = QApplication([])
    window = MainWindow(FLAGS.directory_dir, FLAGS.file_count)
    sys.exit(app.exec_())



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory_dir',
        type=str,
        default='../../../data/vinted/vinted_shirts/',
        help="""\
        Path to the folder of image directories.\
        """
    )

    parser.add_argument(
        '--file_count',
        type=int,
        default=0,
        help="""\
        Number of file in mentioned directory.\
        """
    )

    FLAGS, _ = parser.parse_known_args()
    main(FLAGS)
