"""
Dataset and DataLoader setup using fastai.
"""

from fastai.vision.all import (
    DataBlock, ImageBlock, CategoryBlock,
    get_image_files, RandomSplitter, parent_label, Resize
)

from config import SPECTROGRAM_DIR, VALID_PCT, SEED, IMAGE_SIZE, BATCH_SIZE


def create_dataloaders():
    """Create fastai DataLoaders from spectrogram images."""
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=VALID_PCT, seed=SEED),
        get_y=parent_label,
        item_tfms=[Resize(IMAGE_SIZE, method='squish')]
    ).dataloaders(SPECTROGRAM_DIR, bs=BATCH_SIZE)

    return dls


if __name__ == '__main__':
    dls = create_dataloaders()
    print(f"Training batches: {len(dls.train)}")
    print(f"Validation batches: {len(dls.valid)}")
    print(f"Categories: {dls.vocab}")