# Factory Boy Refactoring Guide

This guide demonstrates how to refactor tests to use factory_boy for cleaner, more maintainable test data creation.

## Benefits of Using Factories

1. **DRY Principle** - Define object creation once, reuse everywhere
2. **Minimal Configuration** - Only specify fields that matter for the test
3. **Realistic Data** - Faker generates realistic test data automatically
4. **Relationships** - Factories can handle complex object relationships
5. **Sequences** - Auto-increment IDs and ensure uniqueness

## Before & After Examples

### Example 1: Creating Training Samples

**BEFORE** (Manual Object Creation):
```python
def test_load_all_samples(self, training_session, label_run):
    samples = [
        TrainingSample(
            label_run_id=label_run.id,
            snapshot_id=1,
            node_id=1,
            name_raw="test1",
            label="variant",
            split="train",
        ),
        TrainingSample(
            label_run_id=label_run.id,
            snapshot_id=1,
            node_id=2,
            name_raw="test2",
            label="subject",
            split="validation",
        ),
        TrainingSample(
            label_run_id=label_run.id,
            snapshot_id=1,
            node_id=3,
            name_raw="test3",
            label=None,
            split="test",
        ),
    ]
    training_session.add_all(samples)
    training_session.commit()

    loaded = load_samples(training_session)
    assert len(loaded) == 3
```

**AFTER** (Using Factories):
```python
def test_load_all_samples(self, training_session, label_run):
    # Create 2 samples with label="variant", other fields use defaults
    TrainingSampleFactory.create_batch(2, label_run_id=label_run.id, label="variant")
    # Create 1 unlabeled sample
    TrainingSampleFactory(label_run_id=label_run.id, label=None)

    loaded = load_samples(training_session)
    assert len(loaded) == 3
```

**Benefits**:
- 15 lines reduced to 6 lines
- Only specify fields relevant to the test (label_run_id, label)
- Factory handles snapshot_id, node_id, name_raw, split with sensible defaults
- create_batch() for creating multiple similar objects

### Example 2: Creating Node Hierarchies

**BEFORE**:
```python
def test_basic_hierarchy(self, index_session, sample_snapshot):
    nodes = [
        Node(
            node_id=1,
            snapshot_id=1,
            name="root",
            kind=NodeKind.DIR,
            parent_node_id=None,
            depth=0,
            rel_path="root",
            abs_path="/test/root",
            ext=None,
            file_source="filesystem",
        ),
        Node(
            node_id=2,
            snapshot_id=1,
            name="child",
            kind=NodeKind.DIR,
            parent_node_id=1,
            depth=1,
            rel_path="root/child",
            abs_path="/test/root/child",
            ext=None,
            file_source="filesystem",
        ),
    ]
    index_session.add_all(nodes)
    index_session.commit()
```

**AFTER**:
```python
def test_basic_hierarchy(self, index_session, sample_snapshot):
    NodeFactory(node_id=1, name="root", depth=0)
    NodeFactory(node_id=2, name="child", parent_node_id=1, depth=1)
    # Factory auto-generates: snapshot_id, kind, rel_path, abs_path, file_source
```

**Benefits**:
- 26 lines reduced to 3 lines
- Only specify what's unique: node_id, name, depth, parent relationship
- Factory infers rel_path and abs_path from name
- Default kind=NodeKind.DIR, file_source="filesystem"

### Example 3: Creating Test Data in Parametrized Tests

**BEFORE**:
```python
@pytest.mark.parametrize("split,expected_count", [
    ("train", 1),
    ("validation", 1),
    ("test", 1),
])
def test_load_samples_by_split(self, training_session, label_run, split, expected_count):
    samples = [
        TrainingSample(
            label_run_id=label_run.id,
            snapshot_id=1,
            node_id=1,
            name_raw="test1",
            label="variant",
            split="train",
        ),
        TrainingSample(
            label_run_id=label_run.id,
            snapshot_id=1,
            node_id=2,
            name_raw="test2",
            label="subject",
            split="validation",
        ),
        TrainingSample(
            label_run_id=label_run.id,
            snapshot_id=1,
            node_id=3,
            name_raw="test3",
            label="other",
            split="test",
        ),
    ]
    training_session.add_all(samples)
    training_session.commit()

    loaded = load_samples(training_session, split=split)
    assert len(loaded) == expected_count
```

**AFTER**:
```python
@pytest.mark.parametrize("split,expected_count", [
    ("train", 1),
    ("validation", 1),
    ("test", 1),
])
def test_load_samples_by_split(self, training_session, label_run, split, expected_count):
    TrainingSampleFactory(label_run_id=label_run.id, split="train")
    TrainingSampleFactory(label_run_id=label_run.id, split="validation")
    TrainingSampleFactory(label_run_id=label_run.id, split="test")

    loaded = load_samples(training_session, split=split)
    assert len(loaded) == expected_count
```

**Benefits**:
- 31 lines reduced to 7 lines
- Clear focus on what's being tested: the split parameter
- Labels are auto-generated with realistic variation via factory.Iterator

## Refactoring Checklist

For each test:

1. ✅ Identify what fields are actually relevant to the test
2. ✅ Replace manual object creation with factory calls
3. ✅ Remove explicit specification of default/unimportant fields
4. ✅ Use `create_batch()` for creating multiple similar objects
5. ✅ Let Faker and Sequence handle varying data
6. ✅ Remove manual session.add() and session.commit() calls (factories handle this)

## Factory Features Used

- `NodeFactory()` - Create single node
- `NodeFactory.create_batch(5)` - Create 5 nodes
- `NodeFactory(name="custom")` - Override specific fields
- `FileNodeFactory()` - Specialized factory for files
- `Faker("word")` - Generate random realistic data
- `Sequence(lambda n: n + 1)` - Auto-increment IDs
- `factory.Iterator([...])` - Cycle through values

## Reproducibility and Seeding

The test suite uses a random seed for factory-generated data to ensure:
- Tests exercise different data variations across runs
- Failed tests can be reproduced by reusing the same seed

**How it works:**
- Each test run generates a random seed (or uses `FACTORY_SEED` env var)
- The seed is printed at the start of the test run
- To reproduce a test failure, set the seed from the output:
  ```bash
  FACTORY_SEED=1234567890 pytest
  ```

**Example output:**
```
======================================================================
Factory seed: 1234567890
To reproduce this test run, set: FACTORY_SEED=1234567890
======================================================================
```

This is configured in `conftest.py` via the `setup_factory_seed` fixture.

## Summary

The factory pattern transforms verbose, repetitive test setup into concise declarations
that focus on what's being tested. This makes tests:

- Easier to read and understand
- Faster to write
- More maintainable when models change
- Less prone to copy-paste errors
