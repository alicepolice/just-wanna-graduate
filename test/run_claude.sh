#!/bin/bash

SESSION="claude-session"

# 创建新的 tmux 会话（后台）
tmux new-session -d -s "$SESSION"

# 启动 claude cli
tmux send-keys -t "$SESSION" "claude" Enter

# 等待 claude 启动
sleep 3

# 发送提示词
tmux send-keys -t "$SESSION" "帮我新建一个文件 hello.txt，内容填入：Hello! " Enter

# 附加到会话，方便查看输出
tmux attach-session -t "$SESSION"
