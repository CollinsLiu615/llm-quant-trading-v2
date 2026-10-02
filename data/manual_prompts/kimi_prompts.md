# 给网页 AI 的 prompts（AI: kimi）

## 操作流程

1. 打开本文件，复制第 1 段 prompt
2. 粘贴给网页 AI，等它输出 JSON
3. **全选 AI 输出的 JSON 并复制**
4. 终端跑下面这条命令保存：
   `pbpaste > ~/Desktop/stock_ai_test_v2/data/news_raw/manual_kimi_1.json`
pbpaste > ~/Desktop/stock_ai_test_v2/data/news_raw/manual_kimi_2.json
pbpaste > ~/Desktop/stock_ai_test_v2/data/news_raw/manual_kimi_3.json
pbpaste > ~/Desktop/stock_ai_test_v2/data/news_raw/manual_kimi_4.json
pbpaste > ~/Desktop/stock_ai_test_v2/data/news_raw/manual_kimi_5.json
5. 段号 1→2→3→4→5，重复以上步骤
6. 全部完成后跑：`python3 scripts/merge_news.py`

---


## 第 1 段：新能源+半导体

```
你是 A 股财经新闻编辑。请为以下 2 个行业各生成 64 条新闻标题：
【新能源】  【半导体】

每个行业覆盖 8 种组合，每种 8 条：
- policy + 利好
- policy + 利空
- breaking + 利好
- breaking + 利空
- sentiment + 利好
- sentiment + 利空
- earnings + 利好
- earnings + 利空

要求：
- 标题 15~40 字，中文
- 提到具体公司名或具体政策/数据，避免"某公司""某行业"
- 风格模仿财联社/证券时报
- 每条附市场冲击估算：magnitude 0.005~0.10，decay 0.3~0.95，duration 3~30 天

只输出 JSON 数组，共 128 条，无 markdown 包裹、无解释。
每条格式：
{"headline": "...", "magnitude": 0.025, "decay": 0.95, "duration": 20,
 "kind": "policy|breaking|sentiment|earnings",
 "industry": "新能源|半导体",
 "direction": "利好|利空"}

```


## 第 2 段：医药+白酒

```
你是 A 股财经新闻编辑。请为以下 2 个行业各生成 64 条新闻标题：
【医药】  【白酒】

每个行业覆盖 8 种组合，每种 8 条：
- policy + 利好
- policy + 利空
- breaking + 利好
- breaking + 利空
- sentiment + 利好
- sentiment + 利空
- earnings + 利好
- earnings + 利空

要求：
- 标题 15~40 字，中文
- 提到具体公司名或具体政策/数据，避免"某公司""某行业"
- 风格模仿财联社/证券时报
- 每条附市场冲击估算：magnitude 0.005~0.10，decay 0.3~0.95，duration 3~30 天

只输出 JSON 数组，共 128 条，无 markdown 包裹、无解释。
每条格式：
{"headline": "...", "magnitude": 0.025, "decay": 0.95, "duration": 20,
 "kind": "policy|breaking|sentiment|earnings",
 "industry": "医药|白酒",
 "direction": "利好|利空"}

```


## 第 3 段：地产+银行

```
你是 A 股财经新闻编辑。请为以下 2 个行业各生成 64 条新闻标题：
【地产】  【银行】

每个行业覆盖 8 种组合，每种 8 条：
- policy + 利好
- policy + 利空
- breaking + 利好
- breaking + 利空
- sentiment + 利好
- sentiment + 利空
- earnings + 利好
- earnings + 利空

要求：
- 标题 15~40 字，中文
- 提到具体公司名或具体政策/数据，避免"某公司""某行业"
- 风格模仿财联社/证券时报
- 每条附市场冲击估算：magnitude 0.005~0.10，decay 0.3~0.95，duration 3~30 天

只输出 JSON 数组，共 128 条，无 markdown 包裹、无解释。
每条格式：
{"headline": "...", "magnitude": 0.025, "decay": 0.95, "duration": 20,
 "kind": "policy|breaking|sentiment|earnings",
 "industry": "地产|银行",
 "direction": "利好|利空"}

```


## 第 4 段：军工+AI算力

```
你是 A 股财经新闻编辑。请为以下 2 个行业各生成 64 条新闻标题：
【军工】  【AI算力】

每个行业覆盖 8 种组合，每种 8 条：
- policy + 利好
- policy + 利空
- breaking + 利好
- breaking + 利空
- sentiment + 利好
- sentiment + 利空
- earnings + 利好
- earnings + 利空

要求：
- 标题 15~40 字，中文
- 提到具体公司名或具体政策/数据，避免"某公司""某行业"
- 风格模仿财联社/证券时报
- 每条附市场冲击估算：magnitude 0.005~0.10，decay 0.3~0.95，duration 3~30 天

只输出 JSON 数组，共 128 条，无 markdown 包裹、无解释。
每条格式：
{"headline": "...", "magnitude": 0.025, "decay": 0.95, "duration": 20,
 "kind": "policy|breaking|sentiment|earnings",
 "industry": "军工|AI算力",
 "direction": "利好|利空"}

```


## 第 5 段：消费+券商

```
你是 A 股财经新闻编辑。请为以下 2 个行业各生成 64 条新闻标题：
【消费】  【券商】

每个行业覆盖 8 种组合，每种 8 条：
- policy + 利好
- policy + 利空
- breaking + 利好
- breaking + 利空
- sentiment + 利好
- sentiment + 利空
- earnings + 利好
- earnings + 利空

要求：
- 标题 15~40 字，中文
- 提到具体公司名或具体政策/数据，避免"某公司""某行业"
- 风格模仿财联社/证券时报
- 每条附市场冲击估算：magnitude 0.005~0.10，decay 0.3~0.95，duration 3~30 天

只输出 JSON 数组，共 128 条，无 markdown 包裹、无解释。
每条格式：
{"headline": "...", "magnitude": 0.025, "decay": 0.95, "duration": 20,
 "kind": "policy|breaking|sentiment|earnings",
 "industry": "消费|券商",
 "direction": "利好|利空"}

```
