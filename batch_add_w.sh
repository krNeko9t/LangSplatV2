# 给列表里的所有文件夹加上拥有者写权限 (u+w)
FOLDERS=("027cd6ea0f" "09d6e808b4" "0a7cc12c0e" "0b031f3119" "0d8ead0038" "116456116b" "17a5e7d36c" "1cefb55d50" "20871b98f3")
BASE_PATH="/mnt/shared-storage-gpfs2/solution-gpfs02/liuyifei/scannet_pick"

for folder in "${FOLDERS[@]}"; do
    chmod u+w "$BASE_PATH/$folder"
done