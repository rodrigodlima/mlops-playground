# MLOps Presentation — Reading Script

**Legenda:** palavras em **negrito** = palavras VIP (desacelere e capriche) · travessão ( — ) = pausa curta · 🖥️ = troca de slide (diga "Next slide, please" ou faça o sinal combinado)

---

## Component three: experiment tracking

🖥️ **[SLIDE 11 — Experiment Tracking: From chaos to clarity]**

Now, this part is very important for **data scientists** — so I want to talk about it **by itself**.

Think about it: when you build a model, you run many experiments. You try different features — different **parameters** — and different algorithms.

And without a tracking system, everything becomes confusing. You have many files with names like "model **final**," "model **final** v2," and — "model **final**... REALLY **final**."

*(pausa — espere a risada)*

I think many people here have files like that.

So, a tracking **tool** like **MLflow** helps solve this problem. Every time you train a model, it saves everything for you: the **parameters** you used — the metrics you got — and the **model file**.

In the end, you have one **table** where you can see **all** your experiments and compare them easily.

And we also get two important things: we can build the same model again — we know how it was created. And a model **registry**: one place to store and manage the different versions of our models.

---

## Component four: tests

🖥️ **[SLIDE 12 — Tests: Code / Data / Model]**

So, in machine learning, we have to test more than regular software.

We still write the normal code tests — is the code correct? Does it work?

But we add two new types of tests.

First, data tests: before training — is the data valid? Is the format right? Are there strange values? Is there too much missing data?

Because bad data creates a bad model — and you don't see it happen. So we check the data at the door.

And second, model tests: before a model goes live — is it really good enough?

Now, we don't just look at the **overall accuracy**. We check the model on important parts of the data — we check it for fairness — and, most important, we compare it against the model already in production.

And that last check is our safety net: the new model must win. If not — the **pipeline** blocks it.

No human has to catch the mistake.

---

## Component five: monitoring

🖥️ **[SLIDE 13 — Monitoring: living cycle diagram]**

And the fifth component: **monitoring**. Now, for me, this is what really makes MLOps different from normal software.

Think about it: a normal app, with no bugs, stays the same forever. But a model is different. Over time, it loses **quality** — all **by itself**.

And this problem has a name — drift.

So, there are two types. Data drift is when the inputs change shape — for example, a new type of customer starts using the product.

And concept drift is when the connection itself changes. Let's say — last year, a **pattern** **meant fraud**. Today — the same **pattern** is normal.

OK — so we monitor on three levels. First, infrastructure: **latency**, memory — is the service even up?

Second, data: do the inputs still look like the **data** we trained on? And third, the model: is the **accuracy**, or the business metric, still good?

Now, when any of these crosses a limit, we fire an alert — and, if possible, the **pipeline** starts a re-train, **by itself**.

And that closes the loop — you can see it here on the slide. In the end, **monitoring** is what makes MLOps a living **cycle** — and not a one-time launch.

---

## The tools and the people

🖥️ **[SLIDE 14 — Tools aren't random]**

OK — let me quickly map the tools, so all these names land somewhere.

They match what we already saw: Git and **DVC** to version code and data — **MLflow** for experiment tracking and the model **registry** — Docker and **Kubernetes** to package and **scale** — and Airflow to run the whole **pipeline**, end to end.

And — just as important — MLOps is not one person's job.

It is the collaboration of the **data scientist**, who builds the model — the ML engineer, who makes it ready for production — and the data engineer, who makes sure that clean data arrives in the first place.

*(pausa — olhe para a plateia)*

In the end, MLOps is a culture — not only a **toolset**.

---

## Cola de pronúncia (palavras VIP)

| Palavra | Como falar |
|---|---|
| final | FAI-nl (como *fine* + l) |
| tool / toolset | TUUL (u longo + língua nos dentes) |
| MLflow | ém-él-FLOU (letras marcadas) |
| DVC | dí-ví-SÍ |
| table | TÊI-bl |
| all / overall | ÓLL (língua nos dentes no final) |
| registry | RÉ-gis-tri (força no começo) |
| parameters | pa-RA-me-ters (força no RA) |
| accuracy | Á-kiu-ra-si (força no começo) |
| monitoring | MÓ-ni-ro-ring (força no começo, flap T) |
| latency | LEI-ten-si (força no começo) |
| quality | KUÓ-li-ri (flap T = r de "cara") |
| data | DEI-ra (flap T) |
| data scientists | DEI-ra SAI-en-tiss (t do meio some) |
| pattern | PÉ-rern (flap T) |
| meant fraud | MÉNT FRÓD (t quase mudo, sem vogal no fim) |
| scale | SKEIL (sem "e" antes do s) |
| Kubernetes | ku-ber-NÉ-tiiz (i longo + z no final) |
| cycle | SAI-kl |
| pipeline | PAI-plain |
