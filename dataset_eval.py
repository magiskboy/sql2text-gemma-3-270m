import click
from datasets import load_dataset
import utils
import matplotlib.pyplot as plt


@click.group()
def main():
    ...


@main.command()
@click.argument(
    'dataset-id',
    type=click.STRING,
)
@click.option(
    '--encoding-name',
    type=click.STRING,
    default='o200k_base',
)
@click.option(
    '--columns',
    type=click.STRING,
    default=None,
)
@click.option(
    '--split',
    type=click.STRING,
    default='train',
)
@click.option(
    '--sensitive/--no-sensitive',
    type=click.BOOL,
    default=False,
)
@click.option(
    '--cross/--no-cross',
    type=click.BOOL,
    default=False,
)
def entropy(
    dataset_id: str,
    encoding_name: str,
    columns: str,
    split: str,
    sensitive: bool,
    cross: bool,
):
    dataset = load_dataset(dataset_id, split=split, cache_dir='/tmp/huggingface/datasets')
    dataset = dataset.filter(lambda x: x['sql_complexity'] == 'single join')
    columns_val = columns.split(',') if columns else None
    if cross:
        tokens, entropy = utils.token_cross_entropy(dataset, encoding_name, columns_val, sensitive) #type:ignore
    else:
        tokens, entropy = utils.token_entropy(dataset, encoding_name, columns_val, sensitive) #type:ignore

    lines = [
        ('Dataset', dataset_id),
        ('Encoding', encoding_name),
        ('Split', split),
        ('Columns', columns),
        ('Sensitive', sensitive),
        ('Tokens', tokens),
    ]
    if cross:
        plt.imshow(utils.log_contrast(entropy, eps=1e-10), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Cross entropy')
        plt.show()
    else:
        lines.append(('Entropy', entropy))

    for label, value in lines:
        print(f'{label:<10}: {value}')

if __name__ == '__main__':
    main()

