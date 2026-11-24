# 版本与更新日志

MoerEngine 使用 `v<major>.<minor>.<patch>` 形式的 Git tag 标记发布，例如 `v0.1.1`。
仓库根目录的 `cliff.toml` 使用 [git-cliff](https://git-cliff.org/) 根据这些 tag 和提交历史生成中文更新日志，不需要 GitHub token 或网络访问。

## 准备工具

Windows 可通过 Scoop 安装：

```powershell
scoop install git-cliff
git-cliff --version
```

其他平台参考 [git-cliff 安装说明](https://git-cliff.org/docs/installation/)。生成前需要完整的提交与 tag 历史；浅克隆可执行 `git fetch --unshallow --tags`，普通克隆执行 `git fetch --tags` 即可。

## 生成更新日志

在仓库根目录运行：

```powershell
git-cliff --config cliff.toml --output CHANGELOG.md
```

输出包含未发布变更及各 tag 对应的版本。配置按功能、修复、性能、重构、文档、测试和构建分组，跳过合并提交；旧的中文描述与普通英文提交会保留在“其他变更”，不会因不符合 Conventional Commits 而丢失。每条记录链接到原始提交，相邻版本之间提供完整差异链接。

只预览未发布部分，不写文件：

```powershell
git-cliff --config cliff.toml --unreleased --strip header
```

## 准备发布

1. 确认构建与相关测试通过，并检查需要说明的不兼容变更。
2. 预览下一个版本号。`git-cliff --config cliff.toml --bumped-version` 根据 Conventional Commits 推导建议；旧格式提交的影响仍需人工判断。
3. 以选定的版本号生成发布说明，例如：

   ```powershell
   git-cliff --config cliff.toml --unreleased --tag v0.2.0 --strip header
   ```

   此处 `--tag` 只为生成结果指定版本，不创建 Git tag。人工整理说明后，再由维护者创建对应 tag 并发布。

生成日志不会修改 `CMakeLists.txt` 中的版本、创建提交、打 tag 或推送远程。发布版本与构建版本需要维护者在发布准备时确认一致。

如果使用 `template.justfile`，对应入口为 `just changelog`、`just release-notes v0.2.0` 和 `just next-version`。`just changelog build/CHANGELOG-preview.md` 可将完整日志写到指定位置。
