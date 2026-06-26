# copy summaries from remote server to local machine
# finds all `run_summary.json` files in the REMOTE_DIR and copies them to LOCAL_DEST

REMOTE="cvcdt004@ramses4.itcc.uni-koeln.de"
REMOTE_DIR="/scratch/cvcdt004"
LOCAL_DEST="./summaries"

mkdir -p "$LOCAL_DEST"

ssh "$REMOTE" "cd $REMOTE_DIR && find . -type f -name 'run_summary.json'" | while read -r filepath; do
  new_name=$(echo "$filepath" | sed 's|^\./||' | tr '/' '_')
  
  echo "Downloading $new_name..."
  scp "$REMOTE:$REMOTE_DIR/${filepath#./}" "$LOCAL_DEST/$new_name"
done
