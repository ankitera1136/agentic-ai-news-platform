from sqlalchemy import Boolean, Column, Integer, String, DateTime, Text, ForeignKey, Table, JSON, LargeBinary, Float
try:
    from sqlalchemy.orm import declarative_base  # SQLAlchemy 2.x
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base  # SQLAlchemy 1.x
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

# Association table for user interests (many-to-many relationship)
user_interests = Table(
    'user_interests',
    Base.metadata,
    Column('user_id', String, ForeignKey('users.id')),
    Column('category_id', Integer, ForeignKey('categories.id'))
)

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Digest preferences
    daily_digest_enabled = Column(Boolean, default=True)
    weekly_digest_enabled = Column(Boolean, default=True)
    instant_notifications = Column(Boolean, default=False)
    digest_time = Column(String, default="09:00")  # HH:MM format
    time_zone = Column(String, default="UTC")
    
    # Relationships
    interests = relationship("Category", secondary=user_interests, back_populates="users")
    subscriptions = relationship("UserSubscription", back_populates="user")

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    keywords = Column(JSON)  # List of keywords for automatic categorization
    
    # Relationships
    users = relationship("User", secondary=user_interests, back_populates="interests")
    articles = relationship("ArticleCategory", back_populates="category")

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(String, primary_key=True)  # Will use filename as ID
    title = Column(String, nullable=False)
    link = Column(String, nullable=False)
    summary = Column(Text, nullable=True)
    llm_summary = Column(Text, nullable=True)
    published = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    source = Column(String, default="TechNews")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Story Clustering columns
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True, index=True)
    is_copy_of = Column(String, ForeignKey("articles.id"), nullable=True, index=True)
    embedding = Column(LargeBinary, nullable=True)
    entities = Column(JSON, nullable=True)
    minhash = Column(LargeBinary, nullable=True)
    text_basis = Column(String, nullable=True)
    clustered_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    categories = relationship("ArticleCategory", back_populates="article")
    cluster = relationship("Cluster", foreign_keys=[cluster_id], back_populates="articles")

class ArticleCategory(Base):
    __tablename__ = "article_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(String, ForeignKey("articles.id"))
    category_id = Column(Integer, ForeignKey("categories.id"))
    relevance_score = Column(Integer, default=1)  # 1-10 scale
    
    # Relationships
    article = relationship("Article", back_populates="categories")
    category = relationship("Category", back_populates="articles")

class UserSubscription(Base):
    __tablename__ = "user_subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    subscription_type = Column(String)  # "daily", "weekly", "instant"
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    keywords = Column(JSON, nullable=True)  # Custom keywords for this subscription
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")

class DigestLog(Base):
    __tablename__ = "digest_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    digest_type = Column(String)  # "daily", "weekly"
    sent_at = Column(DateTime(timezone=True), server_default=func.now())
    article_count = Column(Integer, default=0)
    email_status = Column(String, default="pending")  # "pending", "sent", "failed"
    
class NotificationLog(Base):
    __tablename__ = "notification_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    article_id = Column(String, ForeignKey("articles.id"))
    notification_type = Column(String)  # "instant", "digest"
    sent_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="pending")  # "pending", "sent", "failed"

class Cluster(Base):
    __tablename__ = "clusters"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    primary_article_id = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    size = Column(Integer, default=1, nullable=False)

    articles = relationship("Article", foreign_keys="Article.cluster_id", back_populates="cluster")

class PairCache(Base):
    __tablename__ = "pair_cache"
    
    a_id = Column(String, primary_key=True)
    b_id = Column(String, primary_key=True)
    verdict = Column(String, nullable=False)  # "SAME" | "DIFFERENT"
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class ClusterLog(Base):
    __tablename__ = "cluster_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(String, nullable=False, index=True)
    decision = Column(String, nullable=False)
    best_cluster_id = Column(Integer, nullable=True)
    best_score = Column(Float, nullable=True)
    layer = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class ArticleBody(Base):
    """Cached full article body used only for internal CareerRadar extraction/auditing.

    Never expose body_text through an API or UI.
    """
    __tablename__ = "article_bodies"

    article_id = Column(String, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    body_text = Column(Text, nullable=True)
    fetched_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    status = Column(String, nullable=False, default="failed")  # ok | failed | paywalled | too_short
    http_status = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    attempts = Column(Integer, nullable=False, default=0)


class CareerExtraction(Base):
    """Idempotency ledger for CareerRadar per-article extraction."""
    __tablename__ = "career_extractions"

    article_id = Column(String, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    content_hash = Column(String, nullable=False)
    text_source = Column(String, nullable=False, default="full_body")  # full_body | stored_summary
    status = Column(String, nullable=False, default="pending")  # succeeded | failed
    attempts = Column(Integer, nullable=False, default=0)
    model_used = Column(String, nullable=True)
    signals_count = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)
    extracted_at = Column(DateTime(timezone=True), server_default=func.now())


class CareerSignal(Base):
    """One structured career signal extracted from one article."""
    __tablename__ = "career_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_article_id = Column(
        String, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    company = Column(String, nullable=False, index=True)
    company_raw = Column(String, nullable=True)
    action = Column(String, nullable=False, index=True)
    timing = Column(String, nullable=False, default="current", server_default="current")  # current | past_context | planned
    count = Column(Integer, nullable=True)
    location_city = Column(String, nullable=True)
    location_country = Column(String, nullable=True)
    skills_mentioned = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=False)
    evidence_quote = Column(Text, nullable=True)
    extracted_at = Column(DateTime(timezone=True), server_default=func.now())
