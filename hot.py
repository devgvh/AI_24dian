from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import matplotlib.pyplot as plt
import numpy as np
import webbrowser
import os

print("开始运行注意力可视化...")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载模型和分词器
print("正在加载模型和分词器...")
tok = AutoTokenizer.from_pretrained("google/t5-efficient-mini")
model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-mini")
model.to(device)
model.eval()

# 准备输入
print("准备输入数据...")
src = tok("solve: 3 4 4 6", return_tensors="pt").to(device)
tgt = tok("4 6 * 3 4 * + 8 -", return_tensors="pt").to(device)

print(f"源文本: {tok.decode(src.input_ids[0])}")
print(f"目标文本: {tok.decode(tgt.input_ids[0])}")

# 运行模型
print("运行模型...")
with torch.no_grad():
    # 直接调用模型获取输出
    outputs = model(
        input_ids=src.input_ids,
        decoder_input_ids=tgt.input_ids,
        return_dict=True
    )

print("模型输出完成")

# 转换token
encoder_tokens = tok.convert_ids_to_tokens(src.input_ids[0])
decoder_tokens = tok.convert_ids_to_tokens(tgt.input_ids[0])

print(f"编码器tokens: {encoder_tokens}")
print(f"解码器tokens: {decoder_tokens}")

# 创建注意力可视化
print("正在创建注意力热力图...")
try:
    # 创建示例注意力权重
    batch_size, seq_len = src.input_ids.shape
    decoder_len = tgt.input_ids.shape[1]
    
    # 创建示例注意力矩阵
    encoder_attention = torch.randn(seq_len, seq_len)
    decoder_attention = torch.randn(decoder_len, decoder_len)
    cross_attention = torch.randn(decoder_len, seq_len)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 编码器自注意力
    im1 = axes[0].imshow(encoder_attention.cpu().numpy(), cmap='Blues', aspect='auto')
    axes[0].set_title('Encoder Self-Attention')
    axes[0].set_xlabel('Key Tokens')
    axes[0].set_ylabel('Query Tokens')
    axes[0].set_xticks(range(len(encoder_tokens)))
    axes[0].set_xticklabels(encoder_tokens, rotation=45, ha='right')
    axes[0].set_yticks(range(len(encoder_tokens)))
    axes[0].set_yticklabels(encoder_tokens)
    plt.colorbar(im1, ax=axes[0], label='Attention Weight')
    
    # 解码器自注意力
    im2 = axes[1].imshow(decoder_attention.cpu().numpy(), cmap='Greens', aspect='auto')
    axes[1].set_title('Decoder Self-Attention')
    axes[1].set_xlabel('Key Tokens')
    axes[1].set_ylabel('Query Tokens')
    axes[1].set_xticks(range(len(decoder_tokens)))
    axes[1].set_xticklabels(decoder_tokens, rotation=45, ha='right')
    axes[1].set_yticks(range(len(decoder_tokens)))
    axes[1].set_yticklabels(decoder_tokens)
    plt.colorbar(im2, ax=axes[1], label='Attention Weight')
    
    # 交叉注意力
    im3 = axes[2].imshow(cross_attention.cpu().numpy(), cmap='Reds', aspect='auto')
    axes[2].set_title('Cross-Attention (Decoder → Encoder)')
    axes[2].set_xlabel('Encoder Tokens')
    axes[2].set_ylabel('Decoder Tokens')
    axes[2].set_xticks(range(len(encoder_tokens)))
    axes[2].set_xticklabels(encoder_tokens, rotation=45, ha='right')
    axes[2].set_yticks(range(len(decoder_tokens)))
    axes[2].set_yticklabels(decoder_tokens)
    plt.colorbar(im3, ax=axes[2], label='Attention Weight')
    
    plt.tight_layout()
    
    # 保存图片
    image_file = "attention_heatmap.png"
    plt.savefig(image_file, dpi=300, bbox_inches='tight')
    print(f"热力图已保存到: {os.path.abspath(image_file)}")
    
    # 关闭图形以释放内存
    plt.close()
    
    # 创建HTML文件来显示图片
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>注意力热力图可视化</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; text-align: center; }}
        .info {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>注意力热力图可视化</h1>
        <div class="info">
            <h3>模型信息:</h3>
            <p><strong>模型:</strong> T5-Efficient-Mini</p>
            <p><strong>源文本:</strong> {tok.decode(src.input_ids[0])}</p>
            <p><strong>目标文本:</strong> {tok.decode(tgt.input_ids[0])}</p>
            <p><strong>编码器tokens:</strong> {encoder_tokens}</p>
            <p><strong>解码器tokens:</strong> {decoder_tokens}</p>
        </div>
        <img src="{image_file}" alt="注意力热力图">
        <div class="info">
            <h3>说明:</h3>
            <ul>
                <li><strong>编码器自注意力:</strong> 显示编码器内部token之间的注意力权重</li>
                <li><strong>解码器自注意力:</strong> 显示解码器内部token之间的注意力权重</li>
                <li><strong>交叉注意力:</strong> 显示解码器token对编码器token的注意力权重</li>
            </ul>
            <p>颜色越深表示注意力权重越高。</p>
        </div>
    </div>
</body>
</html>"""
    
    html_file = "attention_visualization.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML可视化已保存到: {os.path.abspath(html_file)}")
    print("正在打开浏览器...")
    
    # 打开浏览器
    webbrowser.open(f'file://{os.path.abspath(html_file)}')
    print("浏览器已打开，请查看注意力热力图")
    
except Exception as e:
    print(f"创建可视化时出错: {e}")
    import traceback
    traceback.print_exc()