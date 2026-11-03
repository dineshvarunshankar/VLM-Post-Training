"""PSC compatibility entrypoint for the shared metadata-only frame measurement."""

from scripts.train.shared.frame_bound import main


if __name__ == '__main__':
    raise SystemExit(main())
