from prepare_big_datasets import prepare_all_big_datasets
from prepare_small_datasets import prepare_all_small_datasets
from argparse import ArgumentParser
if __name__ == '__main__':
    parser = ArgumentParser(
      description="Create lists of image file and label")
    parser.add_argument(
        '--dataset_dir',type=str, default=' ',
        help="Directory of a formatted dataset")
    parser.add_argument(
        '--output_dir', type=str, default=' ',
        help="Output directory for the lists")
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help="Ratio between validation and trainval data. Default 0.2.")
    args = parser.parse_args()
    print(args.dataset_dir)
    prepare_all_small_datasets(args)
    prepare_all_big_datasets()