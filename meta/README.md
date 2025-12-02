# /meta - Meta & Archive Files

**Purpose:** Non-essential files, archives, test assets, and historical documents.

---

## Contents

### Documentation (Reference)
- `PROJECT_STATUS.md` - Status summary from initial setup
- `HOSTING_DISCUSSION.md` - Summary of hosting planning discussion
- `activate_env.sh` - Old environment activation script

### Testing & Validation
- `plantnet_minimal_test.py` - Old minimal test script (reference only)
- `test-image.jpeg` - Test plant image used for validation
- `package_installation.log` - Package installation log

### Data & Metadata
- `plantnet300K_species_id_2_name.json` - PlantNet species ID mappings
- `class_idx_to_species_id.json` - Class index to species ID mappings
- `plantnet300K_metadata.json` - PlantNet metadata
- `datascience_env/` - Old Python environment (archive)

---

## What Not to Use

❌ Don't use `plantnet_minimal_test.py` - it's outdated
❌ Don't activate `activate_env.sh` - use `backend/` environment instead
❌ Don't use `datascience_env/` - it's archived

---

## What to Use Instead

✅ **For testing:** See `backend/` and `frontend/` READMEs
✅ **For metadata:** Copy needed files to `models/` when building
✅ **For reference:** Read `PROJECT_STATUS.md` or `HOSTING_DISCUSSION.md`

---

## When to Reference This Folder

- Looking up species mappings
- Historical context on the project
- Understanding what was tested
- Checking old environment setup

---

**Everything essential is in root, backend/, frontend/, docs/, and models/.**

This folder exists to keep the project root clean.
