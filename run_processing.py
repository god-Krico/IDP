"""Standalone script to process video without the dashboard."""

import sys
from video_processor import process_video


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "video/TLC00001.MP4"

    print("=" * 60)
    print("CASTING YARD - VIDEO PROCESSOR")
    print("=" * 60)

    result = process_video(video_path)

    print("\n" + "=" * 60)
    print("MOULD PROGRESS REPORT")
    print("=" * 60)

    for mould_name, stages in result["timelines"].items():
        print(f"\n  {mould_name}:")
        if not stages:
            print("    No activity detected.")
            continue
        for s in stages:
            print(f"    {s['stage']:25s}  {s['start']}  ->  {s['end']}")

    print(f"\n  Results saved to: output/progress_report.json")
    print(f"  Launch dashboard:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
