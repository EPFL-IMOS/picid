# 1. Remove the "Ghost" modules (that exist on GitLab but not in your file)
git rm --cached local_packages/mantis 2>/dev/null
git rm --cached local_packages/TiViT 2>/dev/null
git rm --cached local_packages/moment 2>/dev/null
git rm --cached local_packages/TimeLLM 2>/dev/null

# 2. Remove the "Misplaced" modules (so we can move them to local_packages)
git rm --cached phmd 2>/dev/null
git rm --cached rul-datasets 2>/dev/null
rm -rf phmd rul-datasets local_packages

# 3. Create the clean folder
mkdir -p local_packages

# 4. Re-add the modules correctly (This fixes the URLs and Paths)
# TabPFN (Needs 'collab_lalr' branch)
git submodule add --force -b collab_lalr https://github.com/EPFL-IMOS/TabPFN-IMOS.git local_packages/TabPFN

# phmd (Needs 'collab_lalr' branch & move to folder)
git submodule add --force -b collab_lalr https://github.com/EPFL-IMOS/imos_phmd.git local_packages/phmd

# rul-datasets (Move to folder)
git submodule add --force https://github.com/tilman151/rul-datasets local_packages/rul-datasets

# TabDPT (SSH)
git submodule add --force git@github.com:layer6ai-labs/TabDPT.git local_packages/TabDPT

# Carte (SSH)
git submodule add --force git@github.com:soda-inria/carte.git local_packages/carte-ai
