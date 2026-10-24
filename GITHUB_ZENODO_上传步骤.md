# GitHub + Zenodo 上传步骤（release/ 已就绪 · tag v1.0）

本地仓已 100% 备好（813 文件 / 74MB / LICENSE / README），只差推送到 GitHub。
无需给任何人令牌，全程约 5 分钟。

## 一、GitHub（3 条命令 + 1 次浏览器登录）
1. 浏览器开 github.com → 登录 → 右上角 + → New repository
   - Repository name 建议: tsfm-distillation-artifacts
   - Public（审稿人可点击，对投稿有利）
   - 其余全部留空（不要勾 README/LICENSE/.gitignore，否则推送冲突）
   - Create repository
2. Git Bash 里依次执行（第 2 步把 <你的用户名> 换成实际用户名，应为 moonzymoon）：
     cd "D:/0科研/工作1/第20篇SCI_A3TSFM蒸馏/release"
     git remote add origin https://github.com/<你的用户名>/tsfm-distillation-artifacts.git
     git push -u origin main --tags
3. 首次推送会弹 GitHub 登录窗口 → 浏览器授权 → 自动开始上传（74MB，几分钟）
4. 完成后把仓库网址发给 ZCode：需要回填进论文数据可用性声明并重编译

## 二、Zenodo DOI（推荐走 GitHub 联动，零额外上传）
1. zenodo.org → Sign in with GitHub（首次需授权）
2. 右侧菜单 GitHub → 找到 tsfm-distillation-artifacts → 开启 On/Off
3. 仓库的 tag v1.0（已打好）会自动归档并生成 DOI
4. DOI 发给 ZCode 回填（录用后正式引用）

## 备选：Zenodo 直接上传（不走 GitHub）
zenodo.org → New upload → 把 release/ 打成 zip 上传 → Reserve DOI
（仅当不想开 GitHub 公开仓时用；审稿人更认 GitHub）
