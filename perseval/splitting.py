"""Automatic split planning for perspectivist datasets.

The protocol has two hard constraints:

* test users never appear in the training split;
* with ``extended=False``, texts in the test split never appear in the training split.

On a sparse dataset (few annotators per text) both hold with a plain split by
user. On a dense one (every annotator sees most texts) every text ends up
annotated by some test user, so the second constraint wipes out the training
split. ``auto_split`` handles both cases with a single knob: a fraction of the
texts is *reserved* for training only (as BREXIT used to do by hand) and the
smallest reserve that brings the test split down to ``target_test_share`` is
chosen automatically.

Everything here works on plain ``(user, text)`` pairs, so it does not depend on
any dataset class.
"""
from dataclasses import dataclass, field
from math import ceil
from random import Random


@dataclass
class SplitPlan:
    train_user_ids: list
    test_user_ids: list
    adaptation_text_ids: list
    test_text_ids: list
    reserved_text_ids: list
    report: dict = field(default_factory=dict)


def auto_split(user_ids, text_ids, user_groups=None,
               test_user_fraction=0.2,
               adaptation_fraction=0.05,
               target_test_share=0.2,
               min_test_annotations=10,
               min_adaptation_annotations=1,
               reserve_grid=None,
               seed=42):
    if reserve_grid is None:
        reserve_grid = [i / 20 for i in range(19)]

    texts_of_user = {}
    for user, text in dict.fromkeys(zip(user_ids, text_ids)):
        texts_of_user.setdefault(user, []).append(text)
    all_users = sorted(texts_of_user)
    all_texts = sorted({t for texts in texts_of_user.values() for t in texts})

    test_users = _sample_test_users(
        all_users, texts_of_user, user_groups, test_user_fraction,
        min_test_annotations + min_adaptation_annotations, Random(seed))

    text_order = list(all_texts)
    Random(seed).shuffle(text_order)

    candidates = [
        _plan_for_reserve(r, text_order, texts_of_user, all_users, test_users,
                          adaptation_fraction, min_test_annotations,
                          min_adaptation_annotations, seed)
        for r in reserve_grid
    ]
    feasible = [p for p in candidates
                if p.test_user_ids and p.report["test_share"] <= target_test_share]
    if feasible:
        plan = feasible[0]
    else:
        usable = [p for p in candidates if p.test_user_ids]
        if not usable:
            raise ValueError(
                "No split leaves any test user with %d test annotations. "
                "Lower min_test_annotations for this dataset." % min_test_annotations)
        plan = min(usable, key=lambda p: p.report["test_share"])

    plan.report.update({
        "strategy": "auto",
        "annotations_per_text": sum(len(t) for t in texts_of_user.values()) / len(all_texts),
        "target_test_share": target_test_share,
        "target_met": plan.report["test_share"] <= target_test_share,
    })
    return plan


def _sample_test_users(all_users, texts_of_user, user_groups, fraction, min_annotations, rng):
    eligible = [u for u in all_users if len(texts_of_user[u]) >= min_annotations]
    if not eligible:
        raise ValueError(
            "No annotator has the %d annotations needed to be a test user." % min_annotations)
    # Keep at least one user out of the test split, otherwise nothing is left to train on.
    n_test = min(max(1, round(fraction * len(all_users))), len(eligible), len(all_users) - 1)

    if not user_groups:
        return set(rng.sample(eligible, n_test))

    by_group = {}
    for user in eligible:
        by_group.setdefault(user_groups[user], []).append(user)
    groups = sorted(by_group, key=str)
    n_test = max(n_test, min(len(groups), len(eligible)))
    # Largest remainder allocation, at least one test user per group when possible.
    quotas = {g: n_test * len(by_group[g]) / len(eligible) for g in groups}
    allocation = {g: min(len(by_group[g]), max(1, int(quotas[g]))) for g in groups}
    for g in sorted(groups, key=lambda g: quotas[g] - int(quotas[g]), reverse=True):
        if sum(allocation.values()) >= n_test:
            break
        if allocation[g] < len(by_group[g]):
            allocation[g] += 1
    test_users = set()
    for g in groups:
        test_users.update(rng.sample(by_group[g], allocation[g]))
    return test_users


def _plan_for_reserve(reserve, text_order, texts_of_user, all_users, sampled_test_users,
                      adaptation_fraction, min_test, min_adaptation, seed):
    rng = Random(seed)
    reserved = set(text_order[:int(reserve * len(text_order))])
    own = {u: [t for t in texts_of_user[u] if t not in reserved] for u in sampled_test_users}

    # Adaptation texts are sampled over *unique* texts, so the fraction holds
    # regardless of how many test users annotated each text.
    candidates = sorted({t for u in sampled_test_users for t in own[u]})
    adaptation = set(rng.sample(candidates, int(len(candidates) * adaptation_fraction)))
    for user in sorted(sampled_test_users):
        missing = min_adaptation - sum(t in adaptation for t in own[user])
        if missing > 0:
            pool = sorted(t for t in own[user] if t not in adaptation)
            adaptation.update(rng.sample(pool, min(missing, len(pool))))

    # A test user's test annotations are their own texts outside the reserve and
    # the adaptation set, so they do not depend on the other test users.
    test_users, dropped = set(), 0
    for user in sampled_test_users:
        n_adaptation = sum(t in adaptation for t in own[user])
        if n_adaptation >= min_adaptation and len(own[user]) - n_adaptation >= min_test:
            test_users.add(user)
        else:
            dropped += 1

    kept_texts = {t for u in test_users for t in own[u]}
    adaptation &= kept_texts
    test_texts = kept_texts - adaptation
    train_users = [u for u in all_users if u not in test_users]

    per_user_test = [sum(t in test_texts for t in own[u]) for u in test_users]
    per_user_adaptation = [sum(t in adaptation for t in own[u]) for u in test_users]
    n_test = sum(per_user_test)
    n_train_extended = sum(len(texts_of_user[u]) for u in train_users)
    n_train_strict = sum(1 for u in train_users for t in texts_of_user[u] if t not in test_texts)

    report = {
        "reserve": reserve,
        "reserved_texts": len(reserved),
        "test_users": len(test_users),
        "test_users_dropped": dropped,
        "train_users": len(train_users),
        "test_annotations": n_test,
        "adaptation_annotations": sum(per_user_adaptation),
        "train_annotations_strict": n_train_strict,
        "train_annotations_extended": n_train_extended,
        "discarded_annotations": sum(len(texts_of_user[u]) - len(own[u]) for u in test_users),
        "test_share": n_test / (n_test + n_train_strict) if n_test + n_train_strict else 1.0,
        "min_test_per_user": min(per_user_test, default=0),
        "min_adaptation_per_user": min(per_user_adaptation, default=0),
    }
    return SplitPlan(
        train_user_ids=train_users,
        test_user_ids=sorted(test_users),
        adaptation_text_ids=sorted(adaptation),
        test_text_ids=sorted(test_texts),
        reserved_text_ids=sorted(reserved),
        report=report,
    )
