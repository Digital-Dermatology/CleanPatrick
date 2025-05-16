# CleanPatrick
CleanPatrick is a large-scale, real-world image data-cleaning benchmark with 496,377 binary annotations from 933 medical crowd workers for ranking off-topic, near-duplicate, and label-error issues.


## Running the Benchmark

Follow these steps to run the evaluation on your CleanPatrick dataset:

1. **Install dependencies or build a container**

   ```bash
   pip install -r requirements.txt
   ```

   OR:
   ```bash
   make run_bash
   ```

2. **Prepare data directories**

   * Place your `CleanPatrick` dataset under `data/CleanPatrick/`.
   * Place the Fitzpatrick17k images under `data/fitzpatrick17k/`.
   * Ensure the CSV metadata file is at `data/fitzpatrick17k/fitzpatrick17k.csv`.

3. **Set up model checkpoints**

   * Pretrained SelfClean checkpoint is available here:
     ```
     models/Fitzpatrick17k/DINO/checkpoint-epoch500.pth
     ```

4. **Run the benchmark script**
   From the project root, execute:

   ```bash
   python -m src.evaluate
   ```

   This will sequentially run the off-topic detection, near-duplicate detection, and label-error detection benchmarks, outputting scores and logs to the console.

5. **Inspect results**

   * Look at the console logs for precision@k, recall@k, AUROC, and AP.

---

*Tips*:

* For additional detectors, plug them into `evaluate_off_topic`, `evaluate_near_duplicates`, or `evaluate_label_errors` following the existing patterns.
