from scrapy.spiders import CrawlSpider,Rule
from scrapy.linkextractors import LinkExtractor
class CrawlingSpider(CrawlSpider):
    name='rescrawler'
    allowed_domains=['arxiv.org']
    start_urls=['https://arxiv.org/']
    rules = (
            # Rule 1: Match arxiv abs pages and call parser
            Rule(
                LinkExtractor(allow=r"/abs/\d{4}\.\d+"),
                callback="parse_item",
                follow=True
            ),

            # Rule 2: Follow all other links within arxiv
            Rule(
                LinkExtractor(allow=r".*"),
                follow=True
            ),
        ) 
    def parse_item(self,response):
        yield{
            'Link':response.url,
            'Title':response.css("h1.title.mathjax::text").get(),
            'Authors':response.css("div.authors a::text").getall(),
            'Description':response.css("blockquote.abstract.mathjax::text").getall()[1].strip(),
            'Category':str(response.css('h1::text').getall()[0].split('>')[0]).strip(),
            'Primary Subject':response.css("span.primary-subject::text").getall()[0].strip(),
            'Subjects': response.css("td.tablecell.subjects::text").getall()[1].split(';') if len(response.css("td.tablecell.subjects::text").getall())>1 else [],
            'Date':response.css("div.dateline::text").get().strip().strip('[').strip(']').strip('Submitted on '),
            'Link of paper':response.css("a#latexml-download-link::attr(href)").get() if response.css("a#latexml-download-link::attr(href)").get() else "",
            'Link of pdf':response.url+response.css("a.abs-button.download-pdf::attr(href)").get()
        }
        