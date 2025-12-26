# Baseline Evaluation Quickstart

Get baseline ML performance metrics in ~30 minutes with minimal labeling effort.

## Why Run Baseline First?

Before investing hours in labeling 800-1200 samples for fine-tuning:

1. ✅ See if the pre-trained model is already good enough
2. ✅ Identify which classes are most difficult
3. ✅ Establish a performance baseline to beat
4. ✅ Understand the value of fine-tuning investment

**Typical Results:**
- Baseline Macro F1: 40-60% (acceptable for some use cases)
- Fine-tuned Macro F1: 82-85+ (significant improvement)

If baseline hits 70%+, you might not need fine-tuning!

---

## 5-Step Baseline Evaluation

### Step 1: Gather Data (2 min)

```bash
uv run python organizer.py gather /path/to/patreon/assets --storage outputs/run
```

This creates a snapshot in `outputs/run/index.db`.

### Step 2: Extract Features (1-5 min)

```bash
uv run python organizer.py model extract-features \
  --index-db outputs/run/index.db \
  --training-db outputs/training.db \
  --snapshot-id 1
```

This populates `training.db` with feature vectors for all folders.

**Time:** ~30 seconds per 1000 folders

### Step 3: Generate Sample CSV (30 sec)

Generate just 200 samples for quick evaluation:

```bash
uv run python organizer.py model generate-samples \
  --index-db outputs/run/index.db \
  --output-csv baseline_eval.csv \
  --snapshot-id 1 \
  --sample-size 200
```

**Why 200?** Enough for statistically significant results, fast to label.

### Step 4: Label Samples (10-20 min)

Open `baseline_eval.csv` in Excel/Google Sheets and fill in the `label` column.

**Valid labels:**
- `primary_author` - Main creator/studio
- `secondary_author` - Collaborator/guest
- `collection` - Campaign/product line
- `subject` - Specific map/asset name
- `media_format` - Technical format (VTT, Print)
- `media_type` - Type (Maps, Tokens, Music)
- `variant` - Variation (Gridded, Night, Clean)
- `other` - Administrative/organizational

**Tips:**
- Use context columns (parent, siblings, children, extensions) to inform your decision
- When unsure, use `other` or skip and delete the row
- Consistency is more important than perfection
- Aim for at least 150-180 labeled samples

**Import labels:**

```bash
uv run python organizer.py training apply-classifications \
  baseline_eval.csv \
  --storage outputs/run
```

### Step 5: Evaluate Baseline Model (1 min)

```bash
uv run python organizer.py model predict \
  --training-db outputs/training.db \
  --use-baseline \
  --taxonomy legacy \
  --labeled-only \
  --output-file baseline_results.csv
```

**Output:**
- Console shows Macro F1-Score, classification report, confusion matrix
- `baseline_results.csv` contains per-sample predictions

---

## Interpreting Results

### Example Output:

```
================================================================================
EVALUATION RESULTS
================================================================================

Total Samples: 185
Accuracy: 0.5946
Macro F1-Score: 0.4823
Weighted F1-Score: 0.5512

--------------------------------------------------------------------------------
CLASSIFICATION REPORT
--------------------------------------------------------------------------------
                    precision    recall  f1-score   support

  primary_author       0.7500    0.4286    0.5455        21
secondary_author       0.4000    0.3636    0.3810        11
      collection       0.5200    0.6500    0.5783        40
         subject       0.5500    0.5833    0.5662        48
    media_format       0.3750    0.3000    0.3333        10
      media_type       0.6500    0.7222    0.6842        36
         variant       0.4500    0.5625    0.5000        16
           other       0.2000    0.3333    0.2500         3

        accuracy                           0.5946       185
       macro avg       0.4869    0.4929    0.4823       185
    weighted avg       0.5729    0.5946    0.5512       185
```

### Decision Guide:

| Macro F1 | Decision |
|----------|----------|
| < 40% | **Fine-tuning strongly recommended** - Baseline struggles with this data |
| 40-60% | **Fine-tuning recommended** - Significant improvement expected (~20-30% gain) |
| 60-70% | **Fine-tuning optional** - Moderate improvement expected (~10-20% gain) |
| > 70% | **Fine-tuning may not be needed** - Baseline is already performing well |

### Look at Confusion Matrix

Identify which classes are confused:

```
MOST COMMON ERRORS
  collection           -> subject              : 8 errors
  subject              -> collection           : 6 errors
  variant              -> media_type           : 4 errors
  media_format         -> media_type           : 3 errors
```

These pairs indicate where fine-tuning will help most.

---

## Next Steps

### If Baseline is Good Enough (Macro F1 > 70%)

You can use the baseline model for production:

```bash
# Apply baseline to all unlabeled data
uv run python organizer.py model predict \
  --training-db outputs/training.db \
  --use-baseline \
  --taxonomy legacy \
  --save-predictions \
  --output-file all_predictions.csv
```

### If Baseline Needs Improvement (Macro F1 < 70%)

Proceed with full fine-tuning workflow:

1. **Label more samples** (800-1200 total)
   ```bash
   uv run python organizer.py model generate-samples \
     --index-db outputs/run/index.db \
     --output-csv training_samples.csv \
     --snapshot-id 1 \
     --sample-size 1000
   ```

2. **Train fine-tuned model**
   ```bash
   uv run python organizer.py model train \
     --data training_samples.csv \
     --output-dir ./models/my_classifier \
     --num-epochs 8
   ```

3. **Compare results** - Fine-tuned should achieve 82-85%+ Macro F1

---

## Time Investment Summary

| Step | Time | Can Skip? |
|------|------|-----------|
| Gather data | 2 min | No |
| Extract features | 1-5 min | No |
| Generate CSV | 30 sec | No |
| Label 200 samples | 10-20 min | No |
| Run baseline eval | 1 min | No |
| **Total** | **~15-30 min** | - |
| | | |
| Label 1000 more samples | 60-120 min | If baseline good |
| Train model | 5-15 min | If baseline good |
| **Full fine-tuning** | **+70-140 min** | **Skip if F1>70%** |

**ROI:** 15-30 minutes of baseline evaluation can save 70-140 minutes of unnecessary fine-tuning!

---

## Troubleshooting

### "No samples found"

Make sure you imported labels:
```bash
uv run python organizer.py training apply-classifications baseline_eval.csv --storage outputs/run
```

### "Model returns random results"

Baseline model has no task-specific training - this is expected! Results should still be better than random (which would be ~12.5% F1 for 8 classes).

### "I want to improve specific classes"

Use `--hardneg-labels` when training to focus on confusable pairs:
```bash
uv run python organizer.py model train \
  --data training.csv \
  --hardneg-labels "collection,subject,variant,media_type"
```

### "Should I use v1/v2 taxonomy instead of legacy?"

For baseline evaluation, stick with `legacy` taxonomy unless you have a specific reason. The proposals v1/v2 are for future fine-tuning experiments.
