# =========================================================
# EVALUATION
# =========================================================

leaderboard = []

mistake_db = {}

stats_db = {}

for name_dir in sorted(SUBMISSIONS_DIR.iterdir()):

    if not name_dir.is_dir():
        continue

    # =========================================================
    # MODEL NAME
    # =========================================================

    model_name = "Unknown"

    pth_files = list(name_dir.glob("*.pth")) + \
                list(name_dir.glob("*.pt"))

    if len(pth_files):
        model_name = pth_files[0].stem

    print(f"\n{'='*50}")
    print("NAME:", name_dir.name)
    print("MODEL:", model_name)
    print(f"{'='*50}")

    try:

        predict = load_predict(name_dir)

        y_true = []
        y_pred = []

        mistakes = []

        total_time = 0

        for class_name in labels:

            class_dir = HIDDEN_DIR / class_name

            if not class_dir.exists():
                continue

            gt = label2id[class_name]

            for img_path in class_dir.iterdir():

                if img_path.suffix.lower() not in [
                    ".jpg", ".jpeg", ".png", ".webp"
                ]:
                    continue

                try:

                    image = Image.open(img_path).convert("RGB")

                    start = time.time()

                    pred = predict(image)

                    total_time += time.time() - start

                    pred = normalize(pred)

                    y_true.append(gt)
                    y_pred.append(pred)

                    if pred != gt:

                        mistakes.append({
                            "image": str(img_path),
                            "true": class_name,
                            "pred": labels[pred - 1]
                        })

                except Exception:

                    print(f"FAILED IMAGE: {img_path.name}")

                    traceback.print_exc()

        # =================================================
        # METRICS
        # =================================================

        acc = accuracy_score(y_true, y_pred) * 100

        f1 = f1_score(
            y_true,
            y_pred,
            average="macro"
        ) * 100

        score = (
            0.9 * acc +
            0.1 * f1
        )

        avg_time = total_time / len(y_true)

        correct = sum(
            int(a == b)
            for a, b in zip(y_true, y_pred)
        )

        print(f"Images       : {len(y_true)}")
        print(f"Accuracy     : {acc:.2f}")
        print(f"Macro F1     : {f1:.2f}")
        print(f"Final Score  : {score:.2f}")
        print(f"Avg Time/img : {avg_time:.4f}s")
        print(f"Mistakes     : {len(mistakes)}")

        leaderboard.append({
            "name": name_dir.name,
            "model": model_name,
            "accuracy": round(acc, 2),
            "macro_f1": round(f1, 2),
            "final_score": round(score, 2),
            "avg_time": round(avg_time, 4)
        })

        # =================================================
        # SAVE DEBUG INFO
        # =================================================

        mistake_db[name_dir.name] = mistakes

        stats_db[name_dir.name] = {
            "total_images": len(y_true),
            "correct": correct,
            "model": model_name
        }

    except Exception:

        print(f"{name_dir.name} FAILED")

        traceback.print_exc()
