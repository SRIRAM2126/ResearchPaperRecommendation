from scrapy.spiders import CrawlSpider,Rule
from scrapy.linkextractors import LinkExtractor
class CrawlingSpider(CrawlSpider):
    name='res3crawler'
    allowed_domains=['arxiv.org']
    start_urls=['https://arxiv.org/']
    rules = (

    # Scrape papers
    Rule(
        LinkExtractor(
            allow=(
                r"/abs/\d{4}\.\d+",
                r"/abs/[a-zA-Z\-.]+/\d+"
            )
        ),
        callback="parse_paper",
        follow=False
    ),

    # Navigate efficiently
    Rule(
        LinkExtractor(
            allow=(
                r"/archive/",
                r"/list/",
                r"/year/"
            )
        ),
        follow=True
    ),
)
    def parse_paper(self,response):
        yield{
            'Link':response.url,
            'Title':response.css("h1.title.mathjax::text").get(),
            'Authors':response.css("div.authors a::text").getall(),
            'Description':response.css("blockquote.abstract.mathjax::text").getall()[1].strip(),
            'Category':response.css('h1::text').getall()[0].split('>')[0].strip() if response.css('h1::text').getall() else "",
            'Primary Subject':response.css("span.primary-subject::text").getall()[0].strip() if response.css("span.primary-subject::text").getall() else "",
            'Subjects': response.css("td.tablecell.subjects::text").getall()[1].split(';')[1:] if len(response.css("td.tablecell.subjects::text").getall())>1 else [],
            'Date':response.css("div.dateline::text").get().strip().strip('[]').strip('Submitted on ').strip('()').strip(),
            'Link of paper':response.css("a#latexml-download-link::attr(href)").get() if response.css("a#latexml-download-link::attr(href)").get() else "",
            'Link of pdf':"https://arxiv.org"+response.css("a.abs-button.download-pdf::attr(href)").get() if response.css("a.abs-button.download-pdf::attr(href)").get() else ""
        }
        