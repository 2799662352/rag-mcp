# 🤝 贡献指南

感谢您对 **Danbooru BGE-M3 RAG Server** 项目的兴趣！我们欢迎所有形式的贡献。

## 🌟 贡献方式

### 🐛 报告Bug
如果您发现了Bug，请：
1. 检查是否已有相关Issue
2. 使用Bug报告模板创建新Issue
3. 提供详细的复现步骤和环境信息

### 💡 功能建议
我们欢迎新功能建议：
1. 先在Discussions中讨论想法
2. 详细描述用例和预期效果
3. 考虑现有架构的兼容性

### 🔧 代码贡献
1. Fork本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交变更：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建Pull Request

## 📋 开发环境设置

### 克隆和安装
```bash
git clone https://github.com/2799662352/rag-mcp.git
cd rag-mcp
pip install -r requirements.txt
```

### 开发依赖
```bash
pip install -e ".[dev]"
```

### 运行测试
```bash
pytest tests/
```

## 🎯 开发规范

### 代码风格
- 使用Python PEP 8标准
- 使用black进行代码格式化：`black .`
- 使用flake8进行代码检查：`flake8`

### 提交规范
使用Conventional Commits格式：
```
类型(范围): 描述

[详细说明]

[footer]
```

**类型示例：**
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建工具或辅助工具的变动

**示例：**
```
feat(search): add BGE-M3 ColBERT vector support

- Implement ColBERT fine-grained matching
- Add vector fusion scoring algorithm
- Improve search accuracy by 12%

Closes #123
```

### 分支策略
- `main`: 稳定的生产代码
- `develop`: 开发分支
- `feature/*`: 功能分支
- `hotfix/*`: 紧急修复分支

## 🧪 测试指南

### 单元测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_search.py

# 生成覆盖率报告
pytest --cov=src tests/
```

### 集成测试
```bash
# 测试BGE-M3模型加载
python tests/test_bge_integration.py

# 测试ChromaDB连接
python tests/test_database.py
```

## 📚 文档贡献

### 文档类型
- **README.md**: 项目概览和快速开始
- **TRAINING.md**: BGE-M3训练指南
- **API文档**: 代码内docstring
- **Wiki**: 详细使用指南

### 文档标准
- 使用清晰的标题结构
- 提供代码示例
- 包含中英文说明
- 保持最新性

## 🔒 安全问题

如果您发现安全漏洞，请**不要**在公开Issue中报告。
请通过以下方式私下联系维护者：
- 邮件：[联系邮箱]
- 使用GitHub Security Advisory

## 📝 Issue模板

### Bug报告
```markdown
**描述**
简要描述Bug的现象

**复现步骤**
1. 执行 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

**预期行为**
描述您期望发生的情况

**实际行为**
描述实际发生的情况

**环境信息**
- OS: [e.g. Ubuntu 22.04]
- Python: [e.g. 3.11]
- PyTorch: [e.g. 2.0.0]
- GPU: [e.g. RTX 4090]

**附加信息**
其他相关信息、日志、截图等
```

### 功能请求
```markdown
**功能描述**
简要描述您希望添加的功能

**用例**
详细描述这个功能的使用场景

**建议实现**
如果有具体的实现想法，请详细描述

**优先级**
- [ ] 高优先级
- [ ] 中优先级  
- [ ] 低优先级
```

## 🎉 贡献者认可

我们会在以下地方认可贡献者：
- README.md中的Contributors部分
- Release Notes中特别感谢
- 项目Wiki的贡献者页面

## 📞 联系方式

- **GitHub Issues**: 技术问题和Bug报告
- **GitHub Discussions**: 功能讨论和问答
- **Wiki**: 详细文档和教程

## 📄 许可证

通过贡献代码，您同意您的贡献将使用与项目相同的MIT许可证。

---

**再次感谢您的贡献！** 🙏

每一个贡献，无论大小，都让这个项目变得更好。