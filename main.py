from app.runner import run_scrapers


def main(hours: int = 24):
    results = run_scrapers(hours=hours)

    print(f"\n=== Scraping Results (last {hours} hours) ===")
    for source, articles in results.items():
        print(f"{source}: {len(articles)} articles")

    return results


if __name__ == "__main__":
    import sys
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    main(hours=hours)