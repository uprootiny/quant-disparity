# Standard Prompt Sets for Evaluation

## Pangram Set (Short)

Semantically equivalent pangram-style sentences for fair comparison.

```json
{
  "en": "The quick brown fox jumps over the lazy dog.",
  "de": "Der schnelle braune Fuchs springt über den faulen Hund.",
  "fr": "Le renard brun rapide saute par-dessus le chien paresseux.",
  "es": "El rápido zorro marrón salta sobre el perro perezoso.",
  "zh": "敏捷的棕色狐狸跳过懒狗。",
  "ja": "素早い茶色の狐が怠け者の犬を飛び越える。",
  "ko": "빠른 갈색 여우가 게으른 개를 뛰어넘는다.",
  "ru": "Быстрая коричневая лиса перепрыгивает через ленивую собаку.",
  "ar": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
  "he": "השועל החום המהיר קופץ מעל הכלב העצלן."
}
```

## Extended Set (Medium ~50 tokens)

For perplexity evaluation with more context.

```json
{
  "en": "The quick brown fox jumps over the lazy dog and runs through the forest looking for food. It is a beautiful sunny day in the countryside.",
  "de": "Der schnelle braune Fuchs springt über den faulen Hund und rennt durch den Wald auf der Suche nach Nahrung. Es ist ein wunderschöner sonniger Tag.",
  "fr": "Le renard brun rapide saute par-dessus le chien paresseux et court dans la forêt à la recherche de nourriture. C'est une belle journée ensoleillée.",
  "zh": "敏捷的棕色狐狸跳过懒狗，穿过森林寻找食物。这是乡村美丽的晴朗日子。",
  "he": "השועל החום המהיר קופץ מעל הכלב העצלן ורץ ביער בחיפוש אחר אוכל. זהו יום שמשי יפה בכפר.",
  "ar": "الثعلب البني السريع يقفز فوق الكلب الكسول ويجري عبر الغابة بحثاً عن الطعام. إنه يوم مشمس جميل."
}
```

## Wikipedia-Style Set (Long ~100 tokens)

From Wikipedia opening paragraphs for realistic text.

```json
{
  "en": "Language is a structured system of communication that consists of grammar and vocabulary. It is the primary means by which humans convey meaning, both in spoken and written forms.",
  "de": "Sprache ist ein strukturiertes Kommunikationssystem, das aus Grammatik und Vokabular besteht. Sie ist das primäre Mittel, mit dem Menschen Bedeutung vermitteln.",
  "zh": "语言是一种由语法和词汇组成的结构化交流系统。它是人类传达意义的主要手段，包括口语和书面形式。",
  "he": "שפה היא מערכת תקשורת מובנית המורכבת מדקדוק ואוצר מילים. היא האמצעי העיקרי שבו בני אדם מעבירים משמעות.",
  "ar": "اللغة هي نظام اتصال منظم يتكون من قواعد ومفردات. وهي الوسيلة الأساسية التي ينقل بها البشر المعنى."
}
```

## Technical Set (Domain-Specific)

For testing domain transfer.

```json
{
  "en": "The transformer architecture revolutionized natural language processing through self-attention mechanisms that capture long-range dependencies.",
  "zh": "变压器架构通过自注意力机制革新了自然语言处理，该机制可以捕获长距离依赖关系。",
  "he": "ארכיטקטורת הטרנספורמר חוללה מהפכה בעיבוד שפה טבעית באמצעות מנגנוני קשב עצמי הלוכדים תלויות ארוכות טווח."
}
```

---

## Usage Guidelines

1. **Perplexity evaluation**: Use Extended Set (medium length)
2. **Quick testing**: Use Pangram Set (short)
3. **Realistic evaluation**: Use Wikipedia-Style Set
4. **Domain transfer**: Use Technical Set

## Token Counts (GPT-2 Tokenizer)

| Language | Pangram | Extended | Wikipedia |
|----------|---------|----------|-----------|
| en | 9 | 32 | 31 |
| de | 12 | 38 | 27 |
| fr | 14 | 40 | 29 |
| zh | 18 | 42 | 45 |
| he | 32 | 78 | 68 |
| ar | 28 | 65 | 54 |

**Observation**: Hebrew and Arabic use ~3x more tokens than English for equivalent text (token fertility gap).

---

*Last updated: 2026-01-04*
