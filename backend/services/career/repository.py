"""Database access helpers for CareerRadar."""

from typing import Iterable, List, Optional

from sqlalchemy.orm import Session

from backend.app.models import Article, ArticleBody, CareerExtraction, CareerSignal


def get_article(db: Session, article_id: str) -> Optional[Article]:
    return db.query(Article).filter(Article.id == article_id).first()


def get_extraction(db: Session, article_id: str) -> Optional[CareerExtraction]:
    return db.query(CareerExtraction).filter(CareerExtraction.article_id == article_id).first()


def get_cached_body(db: Session, article_id: str) -> Optional[ArticleBody]:
    return db.query(ArticleBody).filter(ArticleBody.article_id == article_id).first()


def pending_articles(
    db: Session,
    limit: int,
    cached_only: bool = False,
    offset: int = 0,
) -> List[Article]:
    """Newest-first non-copy articles, optionally restricted to cached bodies."""
    query = db.query(Article).filter(Article.is_copy_of.is_(None))
    if cached_only:
        query = query.join(ArticleBody, ArticleBody.article_id == Article.id).filter(
            ArticleBody.status == "ok",
            ArticleBody.body_text.isnot(None),
        )
    return (
        query.order_by(Article.created_at.desc())
        .offset(max(0, offset))
        .limit(max(1, limit))
        .all()
    )


def replace_signals(db: Session, article_id: str, signals: Iterable[CareerSignal]) -> int:
    """Delete an article's old signals and stage the new ones in the same transaction."""
    db.query(CareerSignal).filter(CareerSignal.source_article_id == article_id).delete(
        synchronize_session=False
    )
    count = 0
    for signal in signals:
        db.add(signal)
        count += 1
    return count


def upsert_extraction(db: Session, article_id: str, **fields) -> CareerExtraction:
    row = get_extraction(db, article_id)
    if row is None:
        row = CareerExtraction(article_id=article_id)
        db.add(row)
    for key, value in fields.items():
        setattr(row, key, value)
    return row


def count_signals(db: Session, article_id: str) -> int:
    return db.query(CareerSignal).filter(CareerSignal.source_article_id == article_id).count()
