from typing import Any, Iterable


def confirmation_state(twitter_items: Iterable[Any], news_items: Iterable[Any]) -> str:
    twitter_list = list(twitter_items)
    news_list = list(news_items)
    if not twitter_list:
        return "no_twitter_event"
    if not news_list:
        return "unconfirmed"
    twitter_tags = {tag for item in twitter_list for tag in item.event_tags}
    news_tags = {tag for item in news_list for tag in item.event_tags}
    if twitter_tags and news_tags and twitter_tags.intersection(news_tags):
        return "confirmed_by_news"
    return "partially_confirmed"
