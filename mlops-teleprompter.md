# 🎤 TELEPROMPTER — MLOps (slides 11–14) · versão 5 min

**Como ler:** fim de linha = respire · linha em branco = pausa real · **negrito** = desacelere e capriche

---

## 🖥️ SLIDE 11 — Experiment Tracking

Component three:
experiment tracking.

When you build a model,
you run many experiments —
different features, **parameters**,
and algorithms.

Without a tracking system,
everything becomes confusing.

You have many files with names like
"model **final**" —
"model **final** v2" —
and... "model **final**... REALLY **final**."

⏸️ *(espere a risada)*

I think many people here
have files like that.

So, a tracking **tool** like **MLflow**
solves this problem.

Every time you train a model,
it saves everything for you:
the **parameters**, the metrics,
and the **model file**.

In the end, you have one **table**
to see and compare
**all** your experiments.

And we also get two important things:

we can build the same model again —

and a model **registry**:
one place to manage
**all** our model versions.

---

## 🖥️ SLIDE 12 — Tests

Component four:
tests.

In machine learning,
we test more than regular software.

We still write the normal code tests.
But we add two new types.

First, data tests:
before training —
is the data valid?
Right format?
Too much missing data?

Because bad data creates a bad model —
so we check the data
at the door.

And second, model tests:
before a model goes live —
is it really good enough?

We don't just look at **overall accuracy**.

We check important parts of the data —
we check fairness —
and, most important,
we compare it against the model
in production.

That last check
is our safety net:

the new model must win.

If not —
the **pipeline** blocks it.

No human
has to catch the mistake.

---

## 🖥️ SLIDE 13 — Monitoring

And the fifth component:
**monitoring**.

For me, this is what really makes MLOps
different from normal software.

A normal app, with no bugs,
stays the same forever.

But a model is different.

Over time, it loses **quality** —
all **by itself**.

And this problem has a name —

drift.

There are two types.

Data drift is when
the inputs change shape —
for example, a new type of customer
starts using the product.

And concept drift is when
the connection itself changes.

Last year, a **pattern** **meant fraud**.

Today —
the same **pattern** is normal.

So we monitor on three levels.

Infrastructure:
**latency**, memory —
is the service even up?

Data:
do the inputs still look like
the **data** we trained on?

And the model:
is the **accuracy** still good?

When any of these crosses a limit,
we fire an alert —
and, if possible,
the **pipeline** starts a re-train,
**by itself**.

That closes the loop
you see on the slide.

In the end,
**monitoring** is what makes MLOps
a living **cycle** —
and not a one-time launch.

---

## 🖥️ SLIDE 14 — Tools & People

OK — let me quickly map the tools.

They match what we already saw:

Git and **DVC**
to version code and data —

**MLflow** for tracking
and the model **registry** —

Docker and **Kubernetes**
to package and **scale** —

and Airflow to run the whole **pipeline**,
end to end.

And — just as important —
MLOps is not one person's job.

It is the collaboration
of the **data scientist**,
who builds the model —

the ML engineer,
who makes it ready for production —

and the data engineer,
who makes sure clean data arrives.

⏸️ *(olhe para a câmera)*

In the end,
MLOps is a culture —
not only a **toolset**.

---

## Cola de pronúncia (revisar antes da call)

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
