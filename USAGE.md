# Usage

## 1. Decrypt secrets

This is only available on haiming's macbook!

```bash
sops -d secrets.enc.yaml > secrets.yaml
```

## 2. Run generation

This is an example on how to run generataion:

```bash
uv run python -m video_gen.harrypotter.transformer \
    --m4b-path ~/Documents/Books/HarryPotterData/3-PrisonerOfAzkaban/PrisonerOfAzkaban.m4b \
    --chapters 15 \
    --output-dir outputs/3-PrisonerOfAzkaban/13-18 \
    --english-reference ~/Documents/Books/HarryPotterData/3-PrisonerOfAzkaban/en.txt \
    --chinese-reference ~/Documents/Books/HarryPotterData/3-PrisonerOfAzkaban/cn.txt
```
