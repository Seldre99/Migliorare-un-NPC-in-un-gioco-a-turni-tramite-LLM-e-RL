import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv("images/dqn/success_rate_model_dqn_800.csv")
df2 = pd.read_csv("images/dqn_llm/success_rate_model_dqn_llm_800.csv")
df3 = pd.read_csv("images/dqn_llm_rl/success_rate_model_dqn_rl_800.csv")

plt.figure(figsize=(10, 6))

plt.plot(df1["Episode"], df1["Success Rate"], label="DQN", color="blue")
plt.plot(df2["Episode"], df2["Success Rate"], label="Helper-LLM", color="green")
plt.plot(df3["Episode"], df3["Success Rate"], label="Helper-LLM + Reviewer-LLM", color="red")

plt.title("Confronto Success Rate per 800 episodi")
plt.xlabel("Episode")
plt.ylabel("Success Rate")

plt.legend()
plt.savefig("Confronto_Success_800.png")
plt.show()


