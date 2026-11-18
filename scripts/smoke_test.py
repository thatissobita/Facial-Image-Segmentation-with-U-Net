"""Execute the actual notebook pipeline on a small local EasyPortrait subset.

Example: python scripts/smoke_test.py --dataset data/smoke-easyportrait
No dataset is downloaded and no physical camera is opened by this script.
"""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    root = Path(__file__).resolve().parents[1]
    for split in ('train', 'val', 'test'):
        if not (dataset / 'images' / split).is_dir():
            parser.error(f'Missing images/{split} under {dataset}')
    run = root / 'outputs' / ('smoke-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    run.mkdir(parents=True)
    os.environ['EASYPORTRAIT_DIR'] = str(dataset)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
    os.chdir(root)
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

    plt.show = lambda: plt.close('all')
    nb = json.loads((root / 'notebooks/01_train_and_evaluate.ipynb').read_text(encoding='utf-8'))
    context = {'__name__': '__main__'}
    start = time.monotonic()
    completed = []
    print(f'Running smoke test; artifacts: {run}', flush=True)
    with (run / 'execution.log').open('w', encoding='utf-8') as log:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            print(f'Executing notebook cell {i}', flush=True)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                # Only shorten training; preserve input size and full architecture.
                if 'EPOCHS = 18' in source:
                    source = source.replace('EPOCHS = 18', 'EPOCHS = 1')
                if 'BATCH_SIZE = 16' in source:
                    source = source.replace('BATCH_SIZE = 16', 'BATCH_SIZE = 1')
                exec(compile(source, f'notebook-cell-{i}', 'exec'), context)
                if 'MODEL_DIR = ROOT' in source:
                    context['MODEL_DIR'] = run / 'models'
                    context['OUTPUT_DIR'] = run
                    context['MODEL_DIR'].mkdir()
                if 'model = build_unet()' in source:
                    initial_weights = context['model'].trainable_weights[0].numpy().copy()
            log.flush()
            completed.append(i)

    model = context['model']
    history = context['history'].history
    assert all(np.isfinite(v).all() for v in history.values()), 'Nonfinite training metrics'
    assert not np.array_equal(initial_weights, model.trainable_weights[0].numpy()), 'Weights did not update'
    images, masks = next(iter(context['test_dataset']))
    assert images.shape[1:] == (256, 256, 3)
    assert int(tf.reduce_min(masks)) >= 0 and int(tf.reduce_max(masks)) <= 8
    pred = model.predict(images, verbose=0)
    assert pred.shape == (1, 256, 256, 9)
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred.sum(axis=-1), 1, atol=1e-5)

    # Check both training-aware deserialization and the webcam's compile=False path.
    checkpoint = context['MODEL_DIR'] / 'unet.keras'
    restored = tf.keras.models.load_model(checkpoint)
    np.testing.assert_allclose(restored.predict(images, verbose=0), pred, rtol=1e-5, atol=1e-6)
    restored.evaluate(context['val_dataset'], verbose=0)
    del restored

    # Execute the webcam notebook against one real image and real model inference.
    # Replace only the camera/UI functions, so no physical camera is accessed.
    import cv2
    frame = cv2.imread(str(next((dataset / 'images/test').glob('*.jpg'))))
    assert frame is not None
    camera_state = {'released': False, 'displayed': False, 'windows_closed': False}

    class ImageCamera:
        def isOpened(self): return True
        def read(self): return True, frame.copy()
        def release(self): camera_state['released'] = True

    def display(title, image):
        assert image.shape == frame.shape
        assert cv2.imwrite(str(run / 'webcam-frame-overlay.png'), image)
        camera_state['displayed'] = True

    patches = {'VideoCapture': lambda index: ImageCamera(), 'imshow': display,
               'waitKey': lambda delay: ord('q'),
               'destroyAllWindows': lambda: camera_state.update(windows_closed=True)}
    original = {key: getattr(cv2, key) for key in patches}
    try:
        for key, value in patches.items(): setattr(cv2, key, value)
        webcam = json.loads((root / 'notebooks/02_webcam_demo.ipynb').read_text(encoding='utf-8'))
        web_context = {'__name__': '__main__'}
        for cell in webcam['cells']:
            if cell['cell_type'] == 'code':
                code = ''.join(cell['source']).replace('ROOT / "models" / "unet.keras"', repr(str(checkpoint)))
                # MODEL_PATH is normally a Path; preserve that type after substitution.
                code = code.replace('MODEL_PATH = ' + repr(str(checkpoint)), 'MODEL_PATH = Path(' + repr(str(checkpoint)) + ')')
                exec(compile(code, 'webcam-notebook', 'exec'), web_context)
    finally:
        for key, value in original.items(): setattr(cv2, key, value)
    assert all(camera_state.values())

    # Confirm that paired augmentation remains executable and preserves mask labels.
    augmented = context['create_dataset']('train', augment_data=True)
    aug_images, aug_masks = next(iter(augmented))
    assert aug_images.shape[1:] == (256, 256, 3)
    assert aug_masks.dtype == tf.uint8
    assert int(tf.reduce_max(aug_masks)) <= 8
    required = ['evaluation.json', 'learning-curves.png', 'segmentation-examples.png', 'overlay-0.png', 'webcam-frame-overlay.png']
    assert all((run / name).is_file() for name in required)
    report = {'status': 'passed', 'tensorflow': tf.__version__,
              'epochs': 1, 'batch_size': 1, 'parameters': model.count_params(),
              'split_sizes': {s: len(list((dataset / 'images' / s).glob('*.jpg'))) for s in ('train','val','test')},
              'completed_training_notebook_cells': completed,
              'checks': ['real JPEG/PNG loading', 'preprocessing', 'full U-Net forward/backward pass',
                         'finite training and validation metrics', 'weights updated', 'checkpoint save/reload',
                         'prediction parity after reload', 'compiled reload evaluation', 'validation/test IoU',
                         'plots and mask overlays', 'augmentation path', 'webcam notebook with real image and model'],
              'physical_webcam_tested': False, 'elapsed_seconds': round(time.monotonic()-start, 2),
              'note': 'Functional smoke test only; metrics are not model-quality estimates.'}
    (run / 'smoke-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print(f'PASS: {run}', flush=True)


if __name__ == '__main__':
    main()
