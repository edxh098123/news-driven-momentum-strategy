### 1. Momentum Signal 与调仓频率

经过对 Traditional 12–1、12–2、Intermediate 12–7、Kalman Filter、Residual Momentum 等多种信号及调仓频率的测试，发现：

- **Intermediate 12–7 在绝大多数实验条件下表现最佳**，具有较明显的收益优势。
- **Annual Rebalancing 整体表现较好**，较长的持仓周期有利于保留持续上涨的股票，同时减少换手和交易成本。
- Annual 对具体调仓月份存在敏感性，但不改变扩展实验中 Intermediate 12–7 的整体优势。

**结论：** 选择 Intermediate 12–7 作为后续重点研究的 Price Momentum Signal，主要保留 Monthly 和 Annual 两种调仓方式。

### 2. Short Leg 持续亏损

原始 Long-Short Momentum Strategy 的 Short Leg 持续亏损，且经过多次调整后仍未获得令人满意的改善。

**结论：** 暂停做空，采用 Long-Only 策略，重点提高长期收益和风险调整后表现。

### 3. Kalman Filter 的效果

对多个 Momentum Window 应用 Kalman Filter，并进行了额外实验验证：

- **Kalman Filter 对 12–1 和 12–2 的优化效果明显**，能够提高趋势平滑性，减少噪声和不必要的换手。
- **Kalman Filter 对 Intermediate 12–7 产生负面效果，部分实验出现亏损。**
- 进一步实验表明，12–7 本身已忽略近期价格变化，叠加 Kalman Filter 后出现过度平滑，导致有效趋势信息损失、信号滞后和投资表现恶化。

**结论：** Kalman Filter 的有效性取决于 Momentum Window。对 Intermediate 12–7 而言，过度平滑造成的信息损失是实验支持的主要解释，因此不再采用 Kalman Filter。

### 4. 不同历史时期的 Momentum 表现

Intermediate 12–7 在 2021–2025 年显著优于 2010–2020 年：

| 时期 | Monthly 净收益 | Annual 净收益 | S&P 500 TR |
|---|---:|---:|---:|
| 2010–2020 | 147.14% | 214.23% | 322.28% |
| 2021–2025 | 178.81% | 262.76% | 96.16% |

**结论：** Intermediate 12–7 具有较强的阶段性盈利能力，但在 2010–2020 年跑输 S&P 500，说明其长期风险收益表现仍有改善空间。

### 5. 收益集中与风险控制

- 2021–2025 年，MU 和 NVDA 合计贡献 Intermediate Annual 五年净收益的约 39%，表明收益高度依赖少数长期赢家。
- Annual 调仓能够保留长期趋势，但可能延迟退出表现恶化的股票。
- 15% Volatility Targeting 虽然降低了实现波动率，但明显牺牲了收益，未能稳定改善最大回撤。

**结论：** 单纯降低仓位不是理想方案，应重点改善 Portfolio Construction 和持仓管理。

### 6. 下一阶段研究

固定 Intermediate 12–7，在 2010–2025 年研究：

1. **Concentration：** Top 10 vs Top 20，检验分散化能否降低个股风险。
2. **Winner Persistence：** 分析退出股票后 1、3、6 个月的收益，研究是否过早卖出赢家。
3. **Drawdown Attribution：** 区分市场风险和个股风险，寻找重大回撤的主要来源。

以 S&P 500 Total Return 和未经优化的 Intermediate 12–7 Annual Top 10 作为比较基准。

**最终研究方向：** 保留已经验证具有优势的 Intermediate 12–7 信号，重点改善长期收益稳定性、个股集中度和退出机制，而不再继续开发新的 Momentum Signal。