"""Optional printed prompts; these are never verified personal handwriting labels."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hwsol.profiles.io import atomic_json
from research.corpus.experiment import create

PROMPTS = {
    "A": [
        "The sample mean changes with observations.",
        "We rewrite the equation before substitution.",
        "The maximum occurs near this boundary.",
        "Different assumptions can produce different conclusions.",
        "This coefficient remains positive throughout integration.",
        "We compare observations with predicted values.",
        "Current follows the indicated arrow direction.",
        "The distance is approximately 20 cm.",
        "Each column represents one independent solution.",
        "The answer includes the appropriate units.",
    ],
    "B": [
        "The parallel components remain approximately unchanged.",
        "Substitute this expression into both equations.",
        "The minimum lies between these endpoints.",
        "The following argument explains this cancellation.",
        "This estimate requires sufficiently large samples.",
        "Adjust the constant using initial conditions.",
        "Probability lies between 0 and 1.",
        "Resistance increases from 2 to 9.",
        "Repeated measurements produce slightly different answers.",
        "The object moves toward the center.",
    ],
    "C": [
        "Symmetry makes the remaining terms vanish.",
        "Temperature changes affect the measured current.",
        "We check this using another method.",
        "Integration gives the total enclosed area.",
        "Both solutions satisfy the boundary conditions.",
        "The average differs from the median.",
        "I obtain approximately 0.29 after rounding.",
        "The projection lies within this plane.",
        "Additional observations provide further independent constraints.",
        "Signs agree with the physical interpretation.",
    ],
}


def build() -> Path:
    run = create(
        "english-continuous-collection-request",
        config={
            "optional_prompts": True,
            "personal_samples_created": 0,
            "C_reserved_for_future_document_holdout": True,
            "prompt_is_not_verified_transcription": True,
            "separate_documents_and_capture_sessions_required": True,
            "prompt_words": 6,
            "fits_existing_1_to_6_word_view_limit": True,
        },
    )
    try:
        records = []
        for session, prompts in PROMPTS.items():
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 595.44)
            ax.set_ylim(0, 841.68)
            ax.axis("off")
            role = (
                "NEW TEST ONLY - do not mix with training"
                if session == "C"
                else "TRAINING MATERIAL"
            )
            ax.text(42, 804, f"Personal English writing / Session {session}", fontsize=15)
            ax.text(42, 783, role, fontsize=10)
            ax.text(
                42, 765, "Write each sentence once, naturally, on the line below it.", fontsize=9
            )
            ax.text(
                42,
                750,
                "Keep this session as a separate PDF. Use your usual pen and writing style.",
                fontsize=8,
            )
            for i, text in enumerate(prompts):
                y = 720 - i * 65
                ax.text(42, y, f"{session}{i + 1:02d}.  {text}", fontsize=9)
                ax.plot([42, 553], [y - 37, y - 37], color="#cccccc", linewidth=0.5)
                records.append(
                    {
                        "id": f"{session}{i + 1:02d}",
                        "session": session,
                        "intended_prompt": text,
                        "verified_text": None,
                        "review_status": "not-collected",
                        "training_eligible": False,
                        "future_split": "holdout" if session == "C" else "train",
                        "answer_bbox_pdf_points": [42, y - 38, 553, y - 6],
                    }
                )
            ax.text(
                42,
                45,
                "Printed prompts are synthetic instructions, not personal training samples.",
                fontsize=8,
            )
            fig.savefig(run.path / f"rendered/session-{session}.pdf")
            fig.savefig(run.path / f"rendered/pages/session-{session}.png", dpi=150)
            plt.close(fig)
        atomic_json(run.path / "reports/prompts.json", records)
        (run.path / "README.txt").write_text(
            "优先提供已有的、包含连续英文解释的真实作业，不必抄模板。\n"
            "若没有，可使用这三份模板，各10句、每句6词，共30行；无需抄写字母表。\n"
            "A/B用于补充训练材料，C只作新文档测试。请在不同书写/采集会话完成并分别保存PDF，不合并。\n"
            "自然书写，不描字体；用常用笔。若整行涂改，另起一行重写，不擦除原记录。\n"
            "扫描约300DPI或清晰正拍，保留四边和颜色，文件放入source_pdf/。\n"
            "也可在空白纸上写，标A/B/C和题号即可。模板文字不自动充当已验证标签，实际笔迹仍须核验。\n"
            "补30行只是下一小试的最低补充建议，不保证模型因此达标；还要修正模型目标与对齐。\n"
        )
        run.finish(
            "prepared",
            result={
                "templates": 3,
                "prompts": 30,
                "real_personal_samples": 0,
                "training_performed": False,
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    print(build())
