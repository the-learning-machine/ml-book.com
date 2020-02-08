# this plugin just creates tags/ files for new tags that are added.
# Once the tags pages are built, add /tags/<pages> to commit
# The script needs to be run first for tags that are added. Run "jekyll build"

Jekyll::Hooks.register :site, :post_write do |post|
  all_existing_tags = Dir.entries("_tags")
    .map { |t| t.match(/(.*).md/) }
    .compact.map { |m| m[1] }

  tags = post.posts.docs.map{ |post| post['tags'] }.flatten.uniq
  tags = tags.reject { |t| t.empty? }

  tags.each do |tag|
    if !all_existing_tags.include?(tag)
      generate_tag_file(tag)
    end
  end
end

def generate_tag_file(tag)
  File.open("_tags/#{tag}.md", "wb") do |file|
    file << "---\n"
    file << "title: #{tag}\n"
    file << "layout: tag-results\n"
    file << "permalink: /tags/#{tag}\n"
    file << "---\n"
  end
end
