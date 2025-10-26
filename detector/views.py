from django.shortcuts import render
from .utils import predict_news

def home(request):
    context = {}

    if request.method == "POST":
        text = request.POST.get("news_text", "")
        if text.strip():
            label, confidence = predict_news(text)
            context = {
                "text": text,
                "label": label,
                "confidence": confidence,
            }

    return render(request, "home.html", context)
